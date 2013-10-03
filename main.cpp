#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <numeric>

#include <QApplication>

#include "common.h"
#include "time_utils.h"
#include "cli_parser.h"
#include "image_loader.h"
#include "NeuralNetworkPixelClassifiers.h"

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

int main(int argc, char **argv)
{
	QApplication app(argc,argv);

	log4cxx::BasicConfigurator::configure(
			log4cxx::AppenderPtr(new log4cxx::ConsoleAppender(
					log4cxx::LayoutPtr(new log4cxx::PatternLayout("\%-5p - [%c] - \%m\%n")),
					log4cxx::ConsoleAppender::getSystemErr()
					)
				)
			);

	log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("main"));

	CliParser cli_parser;
	int parse_result = cli_parser.parse_argv(argc, argv);
	if(parse_result <= 0) {
		exit(parse_result);
	}

	boost::filesystem::path path_export_dir(cli_parser.get_export_dir());

	if(boost::filesystem::exists(path_export_dir)) {
		if(boost::filesystem::is_directory(path_export_dir)) {
			if(!boost::filesystem::is_empty(path_export_dir)) {
				LOG4CXX_FATAL(logger, "Export directory " << path_export_dir << " exists but is not empty");
				exit(-1);
			}
		} else {
			LOG4CXX_FATAL(logger, "Export directory " << path_export_dir << " already exists as a file");
			exit(-1);
		}
	} else {
		if(!boost::filesystem::create_directories(path_export_dir)) {
			LOG4CXX_FATAL(logger, "Export directory " << path_export_dir << " cannot be created");
			exit(-1);
		}
	}

	timestamp_t last_timestamp = get_timestamp();
	LOG4CXX_INFO(logger, "Loading features image");

	typename itk::ImageFileReader< FeaturesImage >::Pointer featuresImageReader = itk::ImageFileReader< FeaturesImage >::New();
	featuresImageReader->SetFileName(cli_parser.get_input_image());

	try {
		featuresImageReader->Update();
	}
	catch( itk::ExceptionObject &ex )
	{
		LOG4CXX_FATAL(logger, "ITK is unable to load the image \"" << cli_parser.get_input_image() << "\" (" << ex.what() << ")");
		exit(-1);
	}

	FeaturesImage::Pointer featuresImage = featuresImageReader->GetOutput();

	LOG4CXX_INFO(logger, "Features image loaded in " << elapsed_time(last_timestamp, get_timestamp()) << "s");


	NeuralNetworkPixelClassifiers pixelClassifiers;

	last_timestamp = get_timestamp();
	LOG4CXX_INFO(logger, "Loading training classes");

	try {
		pixelClassifiers.load_training_sets(cli_parser.get_class_images(), featuresImage);
	} catch (LearningClassException & ex) {
		LOG4CXX_FATAL(logger, "Unable to load the training classes: " << ex.what());
		exit(-1);
	}

	LOG4CXX_INFO(logger, "Training classes loaded in " << elapsed_time(last_timestamp, get_timestamp()) << "s");


	const unsigned int number_of_classifiers = pixelClassifiers.getNumberOfClassifiers();;

    /*
     * Creation of the export folders for each class
     */
	for(int i = 0; i < number_of_classifiers; ++i)
	{
		std::ostringstream export_dir;
		export_dir << cli_parser.get_export_dir() << "/" << std::setfill('0') << std::setw(6) << i;

		boost::filesystem::path path_class_export_dir(export_dir.str());
		if(!boost::filesystem::create_directories(path_class_export_dir)) {
			LOG4CXX_FATAL(logger, "Output dir " << path_class_export_dir << " cannot be created");
			exit(-1);
		}
	}


    /*
     * Training of neural networks
     */
	last_timestamp = get_timestamp();
	LOG4CXX_INFO(logger, "Training neural networks");

	{
		std::vector< unsigned int > hidden_layers = cli_parser.get_ann_hidden_layers();
		hidden_layers.insert(hidden_layers.begin(), featuresImage->GetNumberOfComponentsPerPixel()); // First layer: number of features
		hidden_layers.push_back(1); // Last layer: one output

		pixelClassifiers.create_and_train_neural_networks(hidden_layers, cli_parser.get_ann_learning_rate(), cli_parser.get_ann_max_epoch(), cli_parser.get_ann_mse_target());
	}

	LOG4CXX_INFO(logger, "Neural networks trained in " << elapsed_time(last_timestamp, get_timestamp()) << "s");


	//tlp::initTulipLib("/home/cyrille/Dev/Tulip/tulip-3.8-svn/release/install/");
	LOG4CXX_INFO(logger, "TULIP_DIR set to: " << STRINGIFY(TULIP_DIR));
	//tlp::initTulipLib(STRINGIFY(TULIP_DIR));
	//tlp::initTulipLib(0);
	//tlp::PluginLibraryLoader::loadPlugins(0);
	PluginLoaderTxt txtLoader;
	tlp::initTulipSoftware(&txtLoader);


	/*
	 * Creation of the graph structure
	 */
	last_timestamp = get_timestamp();
	LOG4CXX_INFO(logger, "Generating graph structure");

	tlp::DataSet data;
	data.set("Width",               featuresImage->GetLargestPossibleRegion().GetSize()[0]);
	data.set("Height",              featuresImage->GetLargestPossibleRegion().GetSize()[1]);
	data.set("Depth",               featuresImage->GetLargestPossibleRegion().GetSize()[2]);
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
		std::vector<double> features(featuresImage->GetNumberOfComponentsPerPixel());

		while(itNodes->hasNext())
		{
			u = itNodes->next();
			FeaturesImage::PixelType texture = featuresImage->GetPixel(featuresImage->ComputeIndex(u.id));

			features_tmp = texture.GetDataPointer();
			features.assign(features_tmp, features_tmp + featuresImage->GetNumberOfComponentsPerPixel());
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
		std::ostringstream graph_name;
		graph_name << std::setfill('0') << std::setw(6) << i;

		tlp::Graph* subgraph = graph->addSubGraph(everything, graph_name.str());

		boost::shared_ptr< typename NeuralNetworkPixelClassifiers::NeuralNetwork > net = pixelClassifiers.get_neural_network(i);

		tlp::DoubleProperty *seed = subgraph->getLocalProperty<tlp::DoubleProperty>("Seed");

		tlp::Iterator<tlp::node> *itNodes = subgraph->getNodes();
		tlp::node u;
		tlp::DoubleVectorProperty *f0 = subgraph->getLocalProperty<tlp::DoubleVectorProperty>("f0");
		std::vector<double> features(1);

		while(itNodes->hasNext())
		{
			u = itNodes->next();
			//if(roi->getNodeValue(u)) // TODO Pose problème avec f0_size
			{
				//double* result = fann_run( net.get(), const_cast<fann_type *>( &(features_property->getNodeValue(u)[0]) ) ); // Conversion from vector<double> to double*
				double* result = fann_run( net.get(), const_cast<fann_type *>( features_property->getNodeValue(u).data() ) ); // Conversion from vector<double> to double*
				features[0] = result[0];
				f0->setNodeValue(u, features);
				seed->setNodeValue(u, result[0]); // XXX Changer pour pas de binarisation
			}
		}
		delete itNodes;

		LOG4CXX_INFO(logger, "Data classification done for image #" << i);

		{
			std::ostringstream output_graph;
			output_graph << cli_parser.get_export_dir() << "/graph_" << std::setfill('0') << std::setw(6) << i << ".tlp";
			tlp::saveGraph(subgraph, output_graph.str());
		}


		/*****************************************************/
		/* Application of the graph regularisation algorithm */
		/*****************************************************/
		LOG4CXX_INFO(logger, "Applying CV Regularization algorithm on image #" << i);

		std::ostringstream export_dir;
		export_dir << cli_parser.get_export_dir() << "/" << std::setfill('0') << std::setw(6) << i;

		DoubleProperty* fn = subgraph->getLocalProperty< DoubleProperty >("fn");
		BooleanProperty* segmentation = subgraph->getLocalProperty< BooleanProperty >("viewSelection");

		DataSet data4;
		data4.set("seed",                  seed);
		data4.set("result",                fn);
		data4.set("segmentation result",   segmentation);
		data4.set("data",                  f0);
		data4.set("similarity measure",    weight);
		data4.set("number of iterations",  cli_parser.get_num_iter());
		data4.set("lambda1",               cli_parser.get_lambda1());
		data4.set("lambda2",               cli_parser.get_lambda2());
		data4.set("export interval",       cli_parser.get_export_interval());
		data4.set("dir::export directory", export_dir.str());

		LoggerPluginProgress pp("main.cv_ta");

		string error4;
		if(!subgraph->applyAlgorithm("ChanVese Regularization", error4, &data4, &pp)) {
			LOG4CXX_FATAL(logger, "Unable to apply the ChanVese Regularization algorithm: " << error4);
			return -1;
		}

		regularized_segmentations[i] = fn;

		LOG4CXX_INFO(logger, "Regularization done for image #" << i);
	}

	tlp::saveGraph(graph, cli_parser.get_export_dir() + "/" + "graph.tlp");

	ImageType::Pointer classification_image = ImageType::New();
	classification_image->SetRegions(featuresImage->GetLargestPossibleRegion());
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

	std::string final_export_dir = cli_parser.get_export_dir() + "/final_export";
	if(!boost::filesystem::create_directories(boost::filesystem::path(final_export_dir)))
	{
		LOG4CXX_FATAL(logger, final_export_dir << " cannot be created");
		return -1;
	}

	{
		std::string final_class_export_dir = final_export_dir + "/classmap";
		if(!boost::filesystem::create_directories(boost::filesystem::path(final_class_export_dir)))
		{
			LOG4CXX_FATAL(logger, final_class_export_dir << " cannot be created");
			return -1;
		}

		itk::NumericSeriesFileNames::Pointer outputNames = itk::NumericSeriesFileNames::New();
		final_class_export_dir = final_class_export_dir + "/%06d.bmp";
		outputNames->SetSeriesFormat(final_class_export_dir.c_str());
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
		std::ostringstream class_name;
		if(i > 0)
		{
			class_name << std::setfill('0') << std::setw(6) << i;
		} else {
			class_name << "rejected";
		}

		std::string final_class_export_dir = final_export_dir + "/" + class_name.str();
		if(!boost::filesystem::create_directories(boost::filesystem::path(final_class_export_dir)))
		{
			LOG4CXX_FATAL(logger, final_class_export_dir << " cannot be created");
			return -1;
		}

		itk::NumericSeriesFileNames::Pointer outputNames = itk::NumericSeriesFileNames::New();
		final_class_export_dir = final_class_export_dir + "/%06d.bmp";
		outputNames->SetSeriesFormat(final_class_export_dir.c_str());
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
		//writer->SetSeriesFormat(final_export_dir.c_str());
		writer->SetFileNames(outputNames->GetFileNames());
		writer->Update();
	}

	delete graph;

	return 0;
}
