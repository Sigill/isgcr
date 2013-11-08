#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <cmath>
#include <locale.h>

#include <QApplication>

#include "common.h"
#include "time_utils.h"
#include "cli_parser.h"
#include "image_loader.h"
#include "Classifier.h"
#include "ClassificationDataset.h"
#include "FannClassificationDataset.h"
#include "NeuralNetworkPixelClassifiers.h"
#include "LibSVMClassificationDataset.h"
#include "SVMPixelClassifier.h"

#include "doublefann.h"

#include <tulip/TlpQtTools.h>
#include <tulip/PluginLoaderTxt.h>
#include <tulip/Graph.h>
#include <tulip/TlpTools.h>
#include <tulip/PluginLibraryLoader.h>
#include <tulip/StringCollection.h>
#include <tulip/DoubleProperty.h>
#include <tulip/BooleanProperty.h>

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

class DirException : public std::runtime_error
{
private:
	DirException ( const std::string &err ) : std::runtime_error (err) {}
public:
	static DirException NotEmpty(const bfs::path &path) {     return DirException(path.native() + " is not empty."); }
	static DirException File(const bfs::path &path) {         return DirException(path.native() + " is a file."); }
	static DirException CannotCreate(const bfs::path &path) { return DirException(path.native() + " cannot be created."); }
};

void get_directory(const bfs::path &path, const bool mustBeEmpty = false) {
	if(bfs::exists(path)) {
		if(bfs::is_directory(path)) {
			if(mustBeEmpty && !bfs::is_empty(path))
				throw DirException::NotEmpty(path);
		} else
			throw DirException::File(path);
	} else {
		if(!bfs::create_directories(path))
			throw DirException::CannotCreate(path);
	}
}

std::string pad(const unsigned int i, const char c = '0', const unsigned int l = 6) {
	std::ostringstream os;
	os << std::setfill(c) << std::setw(l) << i;
	return os.str();
}

int main(int argc, char **argv)
{
	QApplication app(argc,argv);
	setlocale(LC_NUMERIC,"C");

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

	boost::shared_ptr< Classifier<fann_type> > pixelClassifier;

	if(cli_parser.get_classifier_training_images_classes().empty()) {
		/*
		 * Loading the classifier from a stored configuration.
		 */
		try {
			if(cli_parser.get_classifier_type() == CliParser::ANN) {
				pixelClassifier = boost::shared_ptr< Classifier<fann_type> >(new NeuralNetworkPixelClassifiers);
			} else if(cli_parser.get_classifier_type() == CliParser::SVM) {
				pixelClassifier = boost::shared_ptr< Classifier<fann_type> >(new SVMPixelClassifier);
			}

			pixelClassifier->load(cli_parser.get_classifier_config_dir());
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

		boost::shared_ptr< ClassificationDataset<double> > trainingDataset;

		try {
			if(cli_parser.get_classifier_training_images().empty()) {
				/*
				 * No image provided for training the classifier, so we
				 * wil use the one that will be segmented.
				 */
				LOG4CXX_INFO(logger, "Loading training classes from input image");

				trainingDataset = boost::shared_ptr< ClassificationDataset<double> >(new ClassificationDataset<double>(input_image, cli_parser.get_classifier_training_images_classes()));
			} else {
				/*
				 * A list of image is available to train the classifier.
				 */
				LOG4CXX_INFO(logger, "Loading training classes from a list of images");

				trainingDataset = boost::shared_ptr< ClassificationDataset<double> >(
						new ClassificationDataset<double>(cli_parser.get_classifier_training_images(), cli_parser.get_classifier_training_images_classes())
				);
			}

			trainingDataset->checkValid();

			LOG4CXX_INFO(logger, "Training classes loaded in " << elapsed_time(last_timestamp, get_timestamp()) << "s");
		} catch (ClassificationDatasetException & ex) {
			LOG4CXX_FATAL(logger, "Unable to load the training classes: " << ex.what());
			exit(-1);
		}

		if(cli_parser.get_classifier_type() == CliParser::ANN)
		{
			/*
			 * Loading the validation classes.
			 */
			last_timestamp = get_timestamp();
			LOG4CXX_INFO(logger, "Loading training classes");

			boost::shared_ptr< ClassificationDataset<double> > validationDataset;

			if(cli_parser.get_ann_validation_images().size() > 0) {
				try {
					/*
					 * A list of image is available to build the validation-set.
					 */
					LOG4CXX_INFO(logger, "Loading validation-classes from a list of images");

					validationDataset = boost::shared_ptr< ClassificationDataset<fann_type> >(
							new ClassificationDataset<double>(cli_parser.get_ann_validation_images(), cli_parser.get_ann_validation_images_classes())
					);

					if(validationDataset->getInputSize() != trainingDataset->getInputSize()) {
						LOG4CXX_FATAL(logger, "The validation set do not have the same number of components per pixel than the training set.");
						exit(-1);
					}

					validationDataset->checkValid();
				} catch (ClassificationDatasetException & ex) {
					LOG4CXX_FATAL(logger, "Unable to load the validation classes: " << ex.what());
					exit(-1);
				}
			} else if(cli_parser.get_ann_validation_training_ratio() > 0) {
				/*
				 * From the training set.
				 */
				LOG4CXX_INFO(logger, "Generating the validation-set from the training-set with a ratio of " << cli_parser.get_ann_validation_training_ratio());

				try {
					trainingDataset->shuffle();

					std::pair< boost::shared_ptr< ClassificationDataset<fann_type> >, boost::shared_ptr< ClassificationDataset<fann_type> > > new_sets =
						trainingDataset->split(cli_parser.get_ann_validation_training_ratio());

					trainingDataset = new_sets.second;
					validationDataset = new_sets.first;

				} catch (FannClassificationDatasetException &ex) {
					LOG4CXX_FATAL(logger, "Cannot generate validation-set from training-set: " << ex.what());
					exit(-1);
				}
			}

			LOG4CXX_INFO(logger, "Validation classes loaded in " << elapsed_time(last_timestamp, get_timestamp()) << "s");

			boost::shared_ptr< FannClassificationDataset > fannTrainingDatasets(new FannClassificationDataset(*trainingDataset)),
			                                               fannValidationDatasets(new FannClassificationDataset(*validationDataset));

			// TODO Supprimer le training set et le validation-set
			// They are not needed now that we have the FannClassificationDatasets
			trainingDataset.reset();
			validationDataset.reset();

			fannTrainingDatasets->shuffle();

			NeuralNetworkPixelClassifiers *ann = new NeuralNetworkPixelClassifiers();
			pixelClassifier = boost::shared_ptr< Classifier<fann_type> >(ann);

			/*
			 * Training of neural networks
			 */
			last_timestamp = get_timestamp();
			LOG4CXX_INFO(logger, "Training neural networks");

			ann->create_neural_networks(fannTrainingDatasets->getInputSize(), fannTrainingDatasets->getNumberOfDatasets(), cli_parser.get_ann_hidden_layers(), cli_parser.get_ann_learning_rate());
			ann->train_neural_networks(fannTrainingDatasets.get(), cli_parser.get_ann_max_epoch(), cli_parser.get_ann_mse_target(), fannValidationDatasets.get());

			LOG4CXX_INFO(logger, "Neural networks trained in " << elapsed_time(last_timestamp, get_timestamp()) << "s");
		} else if(cli_parser.get_classifier_type() == CliParser::SVM) {
			boost::shared_ptr< LibSVMClassificationDataset > svmTrainingDataset(new LibSVMClassificationDataset(*trainingDataset));

			// TODO

			LOG4CXX_FATAL(logger, "Not implemented yet!");
			exit(-1);
		}

		/*
		 * Saving the classifier (if required).
		 */
		if(!cli_parser.get_classifier_config_dir().empty()) {
			bfs::path classifier_config_dir(cli_parser.get_classifier_config_dir());

			try {
				get_directory(classifier_config_dir);
				pixelClassifier->save(cli_parser.get_classifier_config_dir());
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

	if(pixelClassifier->getInputSize() != input_image->GetNumberOfComponentsPerPixel()) {
		LOG4CXX_FATAL(logger, "The classifier is configured to work on pixels with " << pixelClassifier->getInputSize() << " components per pixel, "
		                   << "but the input image has " << input_image->GetNumberOfComponentsPerPixel() << " components per pixel.");
		exit(-1);
	}

	bfs::path export_dir_path(cli_parser.get_export_dir());
	try {
		get_directory(export_dir_path);
	} catch (DirException &err) {
		LOG4CXX_FATAL(logger, err.what());
		exit(-1);
	}

	const unsigned int number_of_classifiers = pixelClassifier->getNumberOfClasses() == 2 ? 1 : pixelClassifier->getNumberOfClasses();

	/*
	 * Creation of the export folders for each class
	 */
	if(cli_parser.get_export_interval() > 0) {
		try {
			for(int i = 0; i < number_of_classifiers; ++i)
				get_directory(export_dir_path / pad(i));
		} catch (DirException &err) {
			LOG4CXX_FATAL(logger, err.what());
			exit(-1);
		}
	}

	//tlp::initTulipLib("/home/cyrille/Dev/Tulip/tulip-3.8-svn/release/install/");
	//LOG4CXX_INFO(logger, "TULIP_DIR set to: " << STRINGIFY(TULIP_DIR));
	//tlp::initTulipLib(STRINGIFY(TULIP_DIR));
	//tlp::initTulipLib(0);
	//tlp::PluginLibraryLoader::loadPlugins(0);
	if(cli_parser.get_debug()) {
		PluginLoaderTxt txtLoader;
		tlp::initTulipSoftware(&txtLoader);
	} else {
		tlp::initTulipSoftware(NULL);
	}


	/*
	 * Creation of the graph structure
	 */
	last_timestamp = get_timestamp();
	LOG4CXX_INFO(logger, "Generating graph structure");

	tlp::DataSet data;
	data.set("Width",               input_image->GetLargestPossibleRegion().GetSize()[0]);
	data.set("Height",              input_image->GetLargestPossibleRegion().GetSize()[1]);
	data.set("Depth",               input_image->GetLargestPossibleRegion().GetSize()[2]);
	data.set("Neighborhood radius", 1.0);
	data.set("Neighborhood type",   tlp::StringCollection("Circular"));
	data.set("Positionning",        true);
	data.set("Spacing",             1.0);

	tlp::Graph *graph = tlp::importGraph("Grid 3D", data);

	tlp::BooleanProperty *everything = graph->getLocalProperty<tlp::BooleanProperty>("everything");
	everything->setAllNodeValue(true);
	everything->setAllEdgeValue(true);

	tlp::BooleanProperty *roi = graph->getLocalProperty<tlp::BooleanProperty>("ROI");

	LOG4CXX_INFO(logger, "Importing region of interest");
	if(cli_parser.get_region_of_interest().empty()) {
		LOG4CXX_INFO(logger, "No region of interest specified");
		roi->setAllNodeValue(true);
	} else {
		tlp::DataSet data;
		std::string error;
		data.set("file::Image",          cli_parser.get_region_of_interest());
		data.set("Property",             roi);
		data.set("Convert to grayscale", false);

		if(!graph->applyAlgorithm("Load image data", error, &data)) {
			LOG4CXX_FATAL(logger, "Unable to import region of interest: " << error);
			return -1;
		}
		LOG4CXX_INFO(logger, "Region of interest successfully imported");
	}


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

	std::vector< tlp::Graph* > subgraphs;
	std::vector< tlp::DoubleProperty* > f0_properties;
	std::vector< tlp::DoubleProperty* > seed_properties;

	/*
	 * Classification of the pixels
	 */
	for(unsigned int i = 0; i < number_of_classifiers; ++i)
	{
		tlp::Graph *subgraph = graph->addSubGraph(everything, pad(i));

		subgraphs.push_back(subgraph);
		seed_properties.push_back(subgraph->getLocalProperty<tlp::DoubleProperty>("Seed"));
		f0_properties.push_back(subgraph->getLocalProperty<tlp::DoubleProperty>("f0"));
	}

	{
		tlp::Iterator<tlp::node> *itNodes = graph->getNodes();
		tlp::node u;
		while(itNodes->hasNext())
		{
			u = itNodes->next();
			if(roi->getNodeValue(u)) // Fut un temps ou cela posait problème avec f0_size, mais cela est réparé
			{
				std::vector<float> probabilities = pixelClassifier->classify(features_property->getNodeValue(u));

				for(unsigned int i = 0; i < number_of_classifiers; ++i)
				{
					f0_properties[i]->setNodeValue(u, probabilities[i]);
					seed_properties[i]->setNodeValue(u, probabilities[i]);
				}
			}
		}
		delete itNodes;
	}

	for(unsigned int i = 0; i < number_of_classifiers; ++i)
	{
		LOG4CXX_INFO(logger, "Data classification done for image #" << i);

		/*****************************************************/
		/* Application of the graph regularisation algorithm */
		/*****************************************************/
		//LOG4CXX_INFO(logger, "Applying CV Regularization algorithm on image #" << i);
		LOG4CXX_INFO(logger, "Applying ROF Regularization algorithm on image #" << i);

		bfs::path export_dir = export_dir_path / pad(i);

		DoubleProperty* fn = subgraphs[i]->getLocalProperty< DoubleProperty >("fn");
		BooleanProperty* segmentation = subgraphs[i]->getLocalProperty< BooleanProperty >("viewSelection");

		DataSet data4;
		data4.set("seed",                  seed_properties[i]);
		data4.set("result",                fn);
		data4.set("segmentation result",   segmentation);
		data4.set("data",                  f0_properties[i]);
		data4.set("similarity measure",    weight);
		data4.set("number of iterations",  cli_parser.get_num_iter());
		data4.set("lambda",               cli_parser.get_lambda1());
		//data4.set("lambda2",               cli_parser.get_lambda2());
		data4.set("export interval",       cli_parser.get_export_interval());
		data4.set("dir::export directory", export_dir.native());

		LoggerPluginProgress pp("main.cv_ta");

		string error4;
		//bool reg_applied = subgraph->applyAlgorithm("ChanVese Regularization", error4, &data4, &pp);
		bool reg_applied = subgraphs[i]->applyAlgorithm("Rudin-Osher-Fatemi Regularization", error4, &data4, &pp);
		if(!reg_applied) {
			LOG4CXX_FATAL(logger, "Unable to apply the ChanVese Regularization algorithm: " << error4);
			return -1;
		}

		regularized_segmentations[i] = fn;

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

				//if( (values[ordered_indices[0]] > 0.5) && ( (0.9 * values[ordered_indices[0]]) > values[ordered_indices[1]]) ) {
				/*
				if( values[ordered_indices[0]] > 0.5 ) {
					max_pos = ordered_indices[0] + 1;
				} else {
					max_pos = 0;
				}
				*/
				max_pos = ordered_indices[0] + 1;
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
		get_directory(final_export_dir_path);
	} catch (DirException &err) {
		LOG4CXX_FATAL(logger, err.what());
		exit(-1);
	}

	{
		bfs::path classmap_export_dir_path = final_export_dir_path / "classmap";
		try {
			get_directory(classmap_export_dir_path);
		} catch (DirException &err) {
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
			get_directory(final_class_export_dir_path);
		} catch (DirException &err) {
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
