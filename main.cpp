#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>

#include "common.h"
#include "time_utils.h"
#include "cli_parser.h"
#include "image_loader.h"
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

	const unsigned int number_of_classes = cli_parser.get_class_images().size();

	// Creation of the export folders for each class
	for(int i = 0; i < number_of_classes; ++i)
	{
		std::ostringstream export_dir;
		export_dir << cli_parser.get_export_dir() << "/" << std::setfill('0') << std::setw(6) << i;

		boost::filesystem::path path_class_export_dir(export_dir.str());
		if(!boost::filesystem::create_directories(path_class_export_dir)) {
			LOG4CXX_FATAL(logger, "Output dir " << path_class_export_dir << " cannot be created");
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


	last_timestamp = get_timestamp();
	LOG4CXX_INFO(logger, "Training neural networks");

	std::vector< unsigned int > hidden_layers = cli_parser.get_ann_hidden_layers();
	hidden_layers.insert(hidden_layers.begin(), featuresImage->GetNumberOfComponentsPerPixel()); // First layer: number of features
	hidden_layers.push_back(1); // Last layer: one output

	pixelClassifiers.create_and_train_neural_networks(hidden_layers, cli_parser.get_ann_learning_rate(), cli_parser.get_ann_max_epoch(), cli_parser.get_ann_mse_target());

	LOG4CXX_INFO(logger, "Neural networks trained in " << elapsed_time(last_timestamp, get_timestamp()) << "s");


	tlp::initTulipLib("/home/cyrille/Dev/Tulip/tulip-3.8-svn/release/install/");
	tlp::loadPlugins(0);


	last_timestamp = get_timestamp();
	LOG4CXX_INFO(logger, "Generating graph structure");

	tlp::DataSet data;
	data.set<int>("Width", featuresImage->GetLargestPossibleRegion().GetSize()[0]);
	data.set<int>("Height", featuresImage->GetLargestPossibleRegion().GetSize()[1]);
	data.set<int>("Depth", featuresImage->GetLargestPossibleRegion().GetSize()[2]);
	data.set<tlp::StringCollection>("Connectivity", tlp::StringCollection("4"));
	data.set<bool>("Positionning", true);
	data.set<double>("Spacing", 1.0);

	tlp::Graph *graph = tlp::importGraph("Grid 3D", data);

	tlp::BooleanProperty *everything = graph->getLocalProperty<tlp::BooleanProperty>("everything");
	everything->setAllNodeValue(true);
	everything->setAllEdgeValue(true);

	tlp::BooleanProperty *roi = graph->getLocalProperty<tlp::BooleanProperty>("Roi");
	roi->setAllNodeValue(true);

	tlp::DoubleProperty *weight = graph->getLocalProperty<tlp::DoubleProperty>("Weight");
	weight->setAllEdgeValue(1);

	tlp::DoubleVectorProperty *features_property = graph->getLocalProperty<tlp::DoubleVectorProperty>("features");

	{
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

	std::vector< tlp::DoubleProperty* > regularized_segmentations(number_of_classes); 

	for(unsigned int i = 0; i < number_of_classes; ++i)
	{
		std::ostringstream graph_name;
		graph_name << std::setfill('0') << std::setw(6) << i;

		tlp::Graph* subgraph = graph->addSubGraph(everything, 0, graph_name.str());

		boost::shared_ptr< typename NeuralNetworkPixelClassifiers::NeuralNetwork > net = pixelClassifiers.get_neural_network(i);

		tlp::BooleanProperty *seed = subgraph->getLocalProperty<tlp::BooleanProperty>("Seed");

		tlp::Iterator<tlp::node> *itNodes = subgraph->getNodes();
		tlp::node u;
		tlp::DoubleVectorProperty *f0 = subgraph->getLocalProperty<tlp::DoubleVectorProperty>("f0");
		std::vector<double> features(1);

		while(itNodes->hasNext())
		{
			u = itNodes->next();
			//double* result = fann_run( net.get(), const_cast<fann_type *>( &(features_property->getNodeValue(u)[0]) ) ); // Conversion from vector<double> to double*
			double* result = fann_run( net.get(), const_cast<fann_type *>( features_property->getNodeValue(u).data() ) ); // Conversion from vector<double> to double*
			features[0] = result[0];
			f0->setNodeValue(u, features);
			seed->setNodeValue(u, result[0] > 0.5);
		}
		delete itNodes;

		LOG4CXX_INFO(logger, "Data classification done for image #" << i);


		{
			std::ostringstream output_graph;
			output_graph << cli_parser.get_export_dir() << "/graph_" << std::setfill('0') << std::setw(6) << i << ".tlp";
			tlp::saveGraph(subgraph, output_graph.str());
		}


		LOG4CXX_INFO(logger, "Applying CV_Ta algorithm on image #" << i);

		std::ostringstream export_dir;
		export_dir << cli_parser.get_export_dir() << "/" << std::setfill('0') << std::setw(6) << i;

		DataSet data4;
		data4.set<PropertyInterface*>("Data", f0);
		data4.set<PropertyInterface*>("Mask", seed);
		data4.set<unsigned int>("Number of iterations", cli_parser.get_num_iter());
		data4.set<double>("Lambda1", cli_parser.get_lambda1());
		data4.set<double>("Lambda2", cli_parser.get_lambda2());
		data4.set<unsigned int>("Export interval", cli_parser.get_export_interval());
		data4.set<string>("dir::Export directory", export_dir.str());
		data4.set<PropertyInterface*>("Weight", weight);
		data4.set<PropertyInterface*>("Roi", roi);

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
	std::vector< double > values(number_of_classes);

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

		for(unsigned int i = 0; i < number_of_classes; ++i)
		{
			values[i] = regularized_segmentations[i]->getNodeValue(u);
		}

		max_it = std::max_element(values.begin(), values.end());

		if(*max_it < 0.9)
			max_pos = 0;
		else
			max_pos = std::distance(values.begin(), max_it) + 1;

		classification_image->SetPixel(index, max_pos);

		//copy( &values[0], &values[networks->size()], std::ostream_iterator< double >(std::cout, ", "));
		//std::cout << std::endl;
	}
	delete itNodes;

	std::string final_export_dir = cli_parser.get_export_dir() + "/final_export";
	if(!boost::filesystem::create_directories(boost::filesystem::path(final_export_dir)))
	{
		std::cerr << final_export_dir << " cannot be created" << std::endl;
		return -1;
	}

	{
		std::string final_class_export_dir = final_export_dir + "/classmap";
		if(!boost::filesystem::create_directories(boost::filesystem::path(final_class_export_dir)))
		{
			std::cerr << final_class_export_dir << " cannot be created" << std::endl;
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

	for(int i = 0; i <= number_of_classes; ++i)
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
			std::cerr << final_class_export_dir << " cannot be created" << std::endl;
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
