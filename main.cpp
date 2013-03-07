#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <cmath>

#include "common.h"
#include "time_utils.h"
#include "cli_parser.h"
#include "image_loader.h"
#include "ClassificationDataset.h"
#include "NeuralNetworkPixelClassifiers.h"

#include "doublefann.h"

#include <tulip/Graph.h>
#include <tulip/TlpTools.h>
#include <tulip/TulipPlugin.h>

#include <boost/filesystem.hpp>

#include <itkImageSeriesWriter.h>
#include <itkNumericSeriesFileNames.h>

#include <itkBinaryThresholdImageFilter.h>

#include "LoggerPluginProgress.h"

#include "log4cxx/logger.h"
#include "log4cxx/consoleappender.h"
#include "log4cxx/patternlayout.h"
#include "log4cxx/basicconfigurator.h"

#include "callgrind.h"

using namespace tlp;
using namespace std;

namespace bfs = boost::filesystem;

//bool desc_comparator(const T a, const T b) { return a > b; }
template <typename T>
class desc_comparator {
public:
	desc_comparator(std::vector<T> const & values) :m_values(values) {}
	inline bool operator() (size_t a, size_t b) { return m_values[a] > m_values[b]; }
private:
	std::vector<T> const& m_values;
};

template <typename T>
std::vector<size_t> ordered(std::vector<T> const& values, desc_comparator<T> comparator) {
	std::vector<size_t> indices(values.size());
	//std::iota(begin(indices), end(indices), static_cast<size_t>(0)); // cxx11
	for (size_t i = 0; i != indices.size(); ++i) indices[i] = i;

	std::sort( indices.begin(), indices.end(), comparator );

	return indices;
}

class EmptyDirException : public std::runtime_error
{
private:
	EmptyDirException ( const std::string &err ) : std::runtime_error (err) {}
public:
	static EmptyDirException NotEmpty(const bfs::path &path) {     return EmptyDirException(path.native() + " " + "is not empty."); }
	static EmptyDirException File(const bfs::path &path) {         return EmptyDirException(path.native() + " " + "is a file."); }
	static EmptyDirException CannotCreate(const bfs::path &path) { return EmptyDirException(path.native() + " " + "cannot be created."); }
};

void get_empty_directory(const bfs::path &path) {
	if(bfs::exists(path)) {
		if(bfs::is_directory(path)) {
			if(!bfs::is_empty(path))
				throw EmptyDirException::NotEmpty(path);
		} else
			throw EmptyDirException::File(path);
	} else {
		if(!bfs::create_directories(path))
			throw EmptyDirException::CannotCreate(path);
	}
}

std::string pad(const unsigned int i, const char c = '0', const unsigned int l = 6) {
	std::ostringstream os;
	os << std::setfill(c) << std::setw(l) << i;
	return os.str();
}

float round(float r) {
	return (r > 0.0) ? floor(r + 0.5) : ceil(r - 0.5);
}

int main(int argc, char **argv)
{
	log4cxx::BasicConfigurator::configure(
			log4cxx::AppenderPtr(new log4cxx::ConsoleAppender(
					log4cxx::LayoutPtr(new log4cxx::PatternLayout("\%-5p - [%c] - \%m\%n")),
					log4cxx::ConsoleAppender::getSystemErr()
					)
				)
			);

	log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("main"));

	CliParser cli_parser;
	try {
		if(cli_parser.parse_argv(argc, argv) != CliParser::CONTINUE)
			exit(0);
	} catch (CliException &err) {
		LOG4CXX_FATAL(logger, err.what());
		return -1;
	}

	timestamp_t last_timestamp;

	FeaturesImage::Pointer input_image;

	/*
	 * Loading the input image (if it exists).
	 */
	if(!cli_parser.get_input_image().empty()) {
		last_timestamp = get_timestamp();
		LOG4CXX_INFO(logger, "Loading features image");

		typename itk::ImageFileReader< FeaturesImage >::Pointer input_image_reader = itk::ImageFileReader< FeaturesImage >::New();
		input_image_reader->SetFileName(cli_parser.get_input_image());

		try {
			input_image_reader->Update();
		} catch( itk::ExceptionObject &ex ) {
			LOG4CXX_FATAL(logger, "ITK is unable to load the image \"" << cli_parser.get_input_image() << "\" (" << ex.what() << ")");
			exit(-1);
		}

		input_image = input_image_reader->GetOutput();

		LOG4CXX_INFO(logger, "Features image loaded in " << elapsed_time(last_timestamp, get_timestamp()) << "s");
	}

	NeuralNetworkPixelClassifiers pixelClassifiers;

	if(cli_parser.get_ann_images_classes().empty()) {
		/*
		 * Loading the neural network from a stored configuration.
		 */
		try {
			pixelClassifiers.load_neural_networks(cli_parser.get_ann_config_dir());
		} catch (std::runtime_error &err) {
			LOG4CXX_FATAL(logger, err.what());
			exit(-1);
		}
	} else {
		/*
		 * Loading the training classes.
		 */
		last_timestamp = get_timestamp();
		LOG4CXX_INFO(logger, "Loading training classes");

		ClassificationDataset trainingDataset;
		boost::shared_ptr< ClassificationDataset::FannDatasetVector > fannTrainingDatasets;

		try {
			if(cli_parser.get_ann_images().empty()) {
				/*
				 * No image provided for training the classifier, so we
				 * wil use the one that will be segmented.
				 */
				LOG4CXX_INFO(logger, "Loading training classes from input image");

				trainingDataset.init(cli_parser.get_ann_images_classes().size());
				trainingDataset.load_image(input_image, cli_parser.get_ann_images_classes());
			} else {
				/*
				 * A list of image is available to train the classifier.
				 */
				std::vector< std::string > ann_images         = cli_parser.get_ann_images(),
				                           ann_images_classes = cli_parser.get_ann_images_classes();

				int number_of_classes = ann_images_classes.size() / ann_images.size();

				trainingDataset.init(number_of_classes);

				for(int i = 0; i < ann_images.size(); ++i) {
					LOG4CXX_INFO(logger, "Loading training image #" << i << " from " << ann_images[i]);

					std::vector< std::string > training_classes(ann_images_classes.begin() + i * number_of_classes, ann_images_classes.begin() + (i+1) * number_of_classes);

					trainingDataset.load_image(ann_images[i], training_classes);
				}
			}

			fannTrainingDatasets = trainingDataset.build_fann_binary_sets();

		} catch (ClassificationDatasetException & ex) {
			LOG4CXX_FATAL(logger, "Unable to load the training classes: " << ex.what());
			exit(-1);
		}

		/*
		 * Shuffling the training datasets.
		 */
		for(int i = 0; i < fannTrainingDatasets->size(); ++i)
			fann_shuffle_train_data(fannTrainingDatasets->operator[](i).get());


		LOG4CXX_INFO(logger, "Training classes loaded in " << elapsed_time(last_timestamp, get_timestamp()) << "s");



		/*
		 * Loading the validation classes.
		 */
		last_timestamp = get_timestamp();
		boost::shared_ptr< ClassificationDataset::FannDatasetVector > fannValidationDatasets;

		if(cli_parser.get_ann_validation_images().size() > 0) {
			/*
			 * From images.
			 */
			LOG4CXX_INFO(logger, "Loading validation classes");

			ClassificationDataset validationDataset;
			validationDataset.init(trainingDataset.getNumberOfClasses());

			try {
				std::vector< std::string > ann_validation_images         = cli_parser.get_ann_validation_images(),
				                           ann_validation_images_classes = cli_parser.get_ann_validation_images_classes();

				for(int i = 0; i < ann_validation_images.size(); ++i) {
					LOG4CXX_INFO(logger, "Loading validation image #" << i << " from " << ann_validation_images[i]);

					std::vector< std::string > validation_classes(
						ann_validation_images_classes.begin() + i * trainingDataset.getNumberOfClasses(),
						ann_validation_images_classes.begin() + (i+1) * trainingDataset.getNumberOfClasses());

					validationDataset.load_image(ann_validation_images[i], validation_classes);

					if((0 == i) && (validationDataset.getDataLength() != trainingDataset.getDataLength())) {
						LOG4CXX_FATAL(logger, "The validation set do not have the same number of components per pixel than the training set.");
						exit(-1);
					}
				}

				fannValidationDatasets = validationDataset.build_fann_binary_sets();

			} catch (ClassificationDatasetException & ex) {
				LOG4CXX_FATAL(logger, "Unable to load the validation classes: " << ex.what());
				exit(-1);
			}
		} else if(cli_parser.get_ann_validation_training_ratio() > 0) {
			/*
			 * From the training set.
			 */
			fannValidationDatasets.reset(new ClassificationDataset::FannDatasetVector(fannTrainingDatasets->size()));

			for(ClassificationDataset::FannDatasetVector::iterator it = fannTrainingDatasets->begin(); it < fannTrainingDatasets->end(); ++it) {
				const unsigned int training_set_size = fann_length_train_data(it->get()),
				                   cut               = round( training_set_size * cli_parser.get_ann_validation_training_ratio() / 100.0f ),
				                   validation_size   = training_set_size - cut;

				if((0 == cut) || (0 == validation_size)) {
					LOG4CXX_FATAL(logger, "The ratio used to generate the validation-set from the training-set ends up generating an empty validation-set.");
					exit(-1);
				}

				it->reset(fann_subset_train_data(it->get(), 0, cut), fann_destroy_train);

				fannValidationDatasets->push_back(boost::shared_ptr< ClassificationDataset::FannDataset >(fann_subset_train_data(it->get(), cut, validation_size), fann_destroy_train));
			}
		}

		LOG4CXX_INFO(logger, "Validation classes loaded in " << elapsed_time(last_timestamp, get_timestamp()) << "s");



		/*
		 * Training of neural networks
		 */
		last_timestamp = get_timestamp();
		LOG4CXX_INFO(logger, "Training neural networks");

		{
			std::vector< unsigned int > ann_layers = cli_parser.get_ann_hidden_layers();

			ann_layers.insert(ann_layers.begin(), trainingDataset.getDataLength()); // First layer: number of features
			ann_layers.push_back(1); // Last layer: one output

			const int numberOfClasses = fannTrainingDatasets->size();
			pixelClassifiers.create_neural_networks((numberOfClasses == 2 ? 1 : numberOfClasses), ann_layers, cli_parser.get_ann_learning_rate());
			pixelClassifiers.train_neural_networks(fannTrainingDatasets, cli_parser.get_ann_max_epoch(), cli_parser.get_ann_mse_target(), fannValidationDatasets);
		}
		// TODO Supprimer le training set et le validation-set

		LOG4CXX_INFO(logger, "Neural networks trained in " << elapsed_time(last_timestamp, get_timestamp()) << "s");

		/*
		 * Sauvegarde de la configuration des réseaux de neurones (si nécessaire).
		 */
		if(!cli_parser.get_ann_config_dir().empty()) {
			bfs::path ann_config_dir(cli_parser.get_ann_config_dir());

			try {
				get_empty_directory(ann_config_dir);
			} catch (EmptyDirException &err) {
				LOG4CXX_FATAL(logger, err.what());
				exit(-1);
			}

			try {
				pixelClassifiers.save_neural_networks(cli_parser.get_ann_config_dir());
			} catch (std::runtime_error &err) {
				LOG4CXX_FATAL(logger, err.what());
				exit(-1);
			}
		}
	}

	/*
	 *
	 *
	 *
	 * The classifier is ready!
	 *
	 *
	 *
	 */

	if(cli_parser.get_input_image().empty())
		exit(0);

	if(pixelClassifiers.getNumberOfComponentsPerPixel() != input_image->GetNumberOfComponentsPerPixel()) {
		LOG4CXX_FATAL(logger, "The classifier is configured to work on pixels with " << pixelClassifiers.getNumberOfComponentsPerPixel() << " components per pixel,");
		LOG4CXX_FATAL(logger, "but the input image has " << input_image->GetNumberOfComponentsPerPixel() << " components per pixel.");
		exit(-1);
	}

	bfs::path export_dir_path(cli_parser.get_export_dir());
	try {
		get_empty_directory(export_dir_path);
	} catch (EmptyDirException &err) {
		LOG4CXX_FATAL(logger, err.what());
		exit(-1);
	}

	const unsigned int number_of_classifiers = pixelClassifiers.getNumberOfClassifiers();

	/*
	 * Creation of the export folders for each class
	 */
	if(cli_parser.get_export_interval() > 0) {
		try {
			for(int i = 0; i < number_of_classifiers; ++i)
				get_empty_directory(export_dir_path / pad(i));
		} catch (EmptyDirException &err) {
			LOG4CXX_FATAL(logger, err.what());
			exit(-1);
		}
	}

	//tlp::initTulipLib("/home/cyrille/Dev/Tulip/tulip-3.8-svn/release/install/");
	LOG4CXX_INFO(logger, "TULIP_DIR set to: " << STRINGIFY(TULIP_DIR));
	tlp::initTulipLib(STRINGIFY(TULIP_DIR));
	tlp::loadPlugins(0);


	/*
	 * Creation of the graph structure
	 */
	last_timestamp = get_timestamp();
	LOG4CXX_INFO(logger, "Generating graph structure");

	tlp::DataSet data;
	data.set<int>("Width", input_image->GetLargestPossibleRegion().GetSize()[0]);
	data.set<int>("Height", input_image->GetLargestPossibleRegion().GetSize()[1]);
	data.set<int>("Depth", input_image->GetLargestPossibleRegion().GetSize()[2]);
	data.set<tlp::StringCollection>("Connectivity", tlp::StringCollection("4"));
	data.set<bool>("Positionning", true);
	data.set<double>("Spacing", 1.0);

	tlp::Graph *graph = tlp::importGraph("Grid 3D", data);

	tlp::BooleanProperty *everything = graph->getLocalProperty<tlp::BooleanProperty>("everything");
	everything->setAllNodeValue(true);
	everything->setAllEdgeValue(true);

	LOG4CXX_INFO(logger, "Importing region of interest");
	if(cli_parser.get_region_of_interest().empty()) {
		LOG4CXX_INFO(logger, "No region of interest specified");
		graph->getLocalProperty<tlp::BooleanProperty>("ROI")->setAllNodeValue(true);
	} else {
		tlp::DataSet data;
		std::string error;
		data.set< string >("dir::Mask folder", cli_parser.get_region_of_interest());
		data.set< StringCollection >("Property type", StringCollection("boolean"));
		data.set< string >("Property name", "ROI");

		if(!graph->applyAlgorithm("Mask for image 3D", error, &data)) {
			LOG4CXX_FATAL(logger, "Unable to import region of interest: " << error);
			return -1;
		}
		LOG4CXX_INFO(logger, "Region of interest imported");
	}

	tlp::BooleanProperty *roi = graph->getLocalProperty<tlp::BooleanProperty>("ROI");

	tlp::DoubleProperty *weight = graph->getLocalProperty<tlp::DoubleProperty>("Weight");
	weight->setAllEdgeValue(1);

	tlp::DoubleVectorProperty *features_property = graph->getLocalProperty<tlp::DoubleVectorProperty>("features");

	{ // Copy of the texture features into the graph
		tlp::Iterator<tlp::node> *itNodes = graph->getNodes();
		tlp::node u;

		const FeaturesImage::PixelType::ValueType *features_tmp;
		std::vector<double> features(input_image->GetNumberOfComponentsPerPixel());

		while(itNodes->hasNext())
		{
			u = itNodes->next();
			FeaturesImage::PixelType texture = input_image->GetPixel(input_image->ComputeIndex(u.id));

			features_tmp = texture.GetDataPointer();
			features.assign(features_tmp, features_tmp + input_image->GetNumberOfComponentsPerPixel());
			features_property->setNodeValue(u, features);
		}
		delete itNodes;
	}

	LOG4CXX_INFO(logger, "Graph structure generated in " << elapsed_time(last_timestamp, get_timestamp()) << "s");


	LOG4CXX_INFO(logger, "Classifying pixels with neural networks");

	std::vector< tlp::DoubleProperty* > regularized_segmentations(number_of_classifiers); 

	/*
	 * Classification of the pixels by each classifier
	 */
	for(unsigned int i = 0; i < number_of_classifiers; ++i)
	{
		tlp::Graph* subgraph = graph->addSubGraph(everything, 0, pad(i));

		boost::shared_ptr< typename NeuralNetworkPixelClassifiers::NeuralNetwork > net = pixelClassifiers.get_neural_network(i);

		tlp::DoubleProperty *seed = subgraph->getLocalProperty<tlp::DoubleProperty>("Seed");

		tlp::Iterator<tlp::node> *itNodes = subgraph->getNodes();
		tlp::node u;
		tlp::DoubleVectorProperty *f0 = subgraph->getLocalProperty<tlp::DoubleVectorProperty>("f0");
		std::vector<double> features(1);

		while(itNodes->hasNext())
		{
			u = itNodes->next();
			if(roi->getNodeValue(u)) // Fut un temps ou cela posait problème avec f0_size, mais cela est réparé
			{
				double* result = fann_run( net.get(), const_cast<fann_type *>( features_property->getNodeValue(u).data() ) ); // Conversion from vector<double> to double*
				features[0] = result[0];
				f0->setNodeValue(u, features);
				seed->setNodeValue(u, result[0]);
			}
		}
		delete itNodes;

		LOG4CXX_INFO(logger, "Data classification done for image #" << i);

		/*****************************************************/
		/* Application of the graph regularisation algorithm */
		/*****************************************************/
		LOG4CXX_INFO(logger, "Applying CV_Ta algorithm on image #" << i);

		bfs::path export_dir = export_dir_path / pad(i);

		DataSet data4;
		data4.set<PropertyInterface*>("data", f0);
		data4.set<PropertyInterface*>("seed", seed);
		data4.set<unsigned int>("number of iterations", cli_parser.get_num_iter());
		data4.set<double>("lambda1", cli_parser.get_lambda1());
		data4.set<double>("lambda2", cli_parser.get_lambda2());
		data4.set<unsigned int>("export interval", cli_parser.get_export_interval());
		data4.set<string>("dir::export directory", export_dir.native());
		data4.set<PropertyInterface*>("weight", weight);
		data4.set<PropertyInterface*>("region of interest", roi);

		LoggerPluginProgress pp("main.cv_ta");

		string error4;
		if(!subgraph->applyAlgorithm("Cv_Ta", error4, &data4, &pp)) {
			LOG4CXX_FATAL(logger, "Unable to apply the Cv_Ta algorithm: " << error4);
			return -1;
		}

		subgraph->delLocalProperty("Seed");

		regularized_segmentations[i] = subgraph->getLocalProperty< DoubleProperty >("fn");

		LOG4CXX_INFO(logger, "Regularization done for image #" << i);
	}

	if(cli_parser.get_debug()) {
		bfs::path output_graph = export_dir_path / "graph.tlp";
	}

	ImageType::Pointer classification_image = ImageType::New();
	classification_image->SetRegions(input_image->GetLargestPossibleRegion());
	classification_image->Allocate();
	ImageType::IndexType index;

	int width, height, depth;
	unsigned int id;

	graph->getAttribute<int>("width", width);
	graph->getAttribute<int>("height", height);
	graph->getAttribute<int>("depth", depth);

	tlp::Iterator<tlp::node> *itNodes = graph->getNodes();
	tlp::node u;
	std::vector< double > values(number_of_classifiers);

	std::vector< double >::iterator max_it;
	unsigned int max_pos;

	while(itNodes->hasNext())
	{
		u = itNodes->next();
		id = u.id;
		index[0] =  id % width;
		id /= width;
		index[1] = id % height;
		id /= height;
		index[2] = id;

		if(roi->getNodeValue(u))
		{
			if(number_of_classifiers > 1)
			{
				for(unsigned int i = 0; i < number_of_classifiers; ++i)
				{
					values[i] = regularized_segmentations[i]->getNodeValue(u);
				}

				std::vector< size_t > ordered_indices = ordered(values, desc_comparator<double>(values));

				if( (values[ordered_indices[0]] > 0.5) && ( (0.9 * values[ordered_indices[0]]) > values[ordered_indices[1]]) ) {
					max_pos = ordered_indices[0] + 1;
				} else {
					max_pos = 0;
				}
			} else {
				max_pos = (regularized_segmentations[0]->getNodeValue(u) > 0.5 ? 1 : 2); // No rejected class
			}
		} else {
			max_pos = 0; // XXX Est ce que l'on met à 0 lorsque l'on est sensé ignorer le pixel ?
		}

		classification_image->SetPixel(index, max_pos);

		//copy( &values[0], &values[networks->size()], std::ostream_iterator< double >(std::cout, ", "));
		//std::cout << std::endl;
	}
	delete itNodes;

	bfs::path final_export_dir_path = export_dir_path / "final_export";

	try {
		get_empty_directory(final_export_dir_path);
	} catch (EmptyDirException &err) {
		LOG4CXX_FATAL(logger, err.what());
		exit(-1);
	}

	{
		bfs::path classmap_export_dir_path = final_export_dir_path / "classmap";
		try {
			get_empty_directory(classmap_export_dir_path);
		} catch (EmptyDirException &err) {
			LOG4CXX_FATAL(logger, err.what());
			exit(-1);
		}

		bfs::path export_dir_pattern = classmap_export_dir_path / "%06d.bmp";
		itk::NumericSeriesFileNames::Pointer outputNames = itk::NumericSeriesFileNames::New();
		outputNames->SetSeriesFormat(export_dir_pattern.native());
		outputNames->SetStartIndex(0);
		outputNames->SetEndIndex(depth - 1);

		typedef itk::ImageSeriesWriter< ImageType, itk::Image< unsigned char, 2 > > WriterType;
		WriterType::Pointer writer = WriterType::New();
		writer->SetInput(classification_image);
		writer->SetFileNames(outputNames->GetFileNames());
		writer->Update();
	}

	for(int i = 0; i <= number_of_classifiers; ++i)
	{
		bfs::path final_class_export_dir_path = final_export_dir_path / (i == 0 ? "rejected" : pad(i));

		try {
			get_empty_directory(final_class_export_dir_path);
		} catch (EmptyDirException &err) {
			LOG4CXX_FATAL(logger, err.what());
			exit(-1);
		}

		bfs::path final_class_export_dir_pattern = final_class_export_dir_path / "%06d.bmp";
		itk::NumericSeriesFileNames::Pointer outputNames = itk::NumericSeriesFileNames::New();
		outputNames->SetSeriesFormat(final_class_export_dir_pattern.native());
		outputNames->SetStartIndex(0);
		outputNames->SetEndIndex(depth - 1);

		typedef itk::BinaryThresholdImageFilter< ImageType, ImageType > Thresholder;
		Thresholder::Pointer thresholder = Thresholder::New();
		thresholder->SetLowerThreshold(i);
		thresholder->SetUpperThreshold(i);
		thresholder->SetInput(classification_image);

		typedef itk::ImageSeriesWriter< ImageType, itk::Image< unsigned char, 2 > > WriterType;
		WriterType::Pointer writer = WriterType::New();
		writer->SetInput(thresholder->GetOutput());
		writer->SetFileNames(outputNames->GetFileNames());
		writer->Update();
	}

	delete graph;

	return 0;
}
