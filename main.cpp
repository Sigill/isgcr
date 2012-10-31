#include <iostream>
#include <vector>
#include <algorithm>

#include "common.h"
#include "time_utils.h"
#include "cli_parser.h"
#include "classification.h"
#include "image_loader.h"

#include "haralick.h"

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

	// Creation of the export folders for each class
	for(int i = 0; i < cli_parser.get_class_images().size(); ++i)
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
	LOG4CXX_INFO(logger, "Computing Haralick features");

	NormalizedHaralickImage::Pointer haralickImage = load_texture_image(cli_parser.get_input_image(), cli_parser.get_num_gray(), cli_parser.get_window_radius(), cli_parser.get_offset());

	LOG4CXX_INFO(logger, "Haralick features computed in " << elapsed_time(last_timestamp, get_timestamp()) << "s");


	last_timestamp = get_timestamp();
	LOG4CXX_INFO(logger, "Loading training classes");

	boost::shared_ptr< TrainingClassVector > training_classes;
	try {
		training_classes = load_classes(cli_parser.get_class_images(), haralickImage);
	} catch (LearningClassException & ex) {
		LOG4CXX_FATAL(logger, "Unable to load the training classes: " << ex.what());
		exit(-1);
	}

	LOG4CXX_INFO(logger, "Training classes loaded in " << elapsed_time(last_timestamp, get_timestamp()) << "s");


	last_timestamp = get_timestamp();
	LOG4CXX_INFO(logger, "Generating training sets");

	boost::shared_ptr< TrainingSetVector > training_sets = generate_training_sets(training_classes);

	LOG4CXX_INFO(logger, "Training sets generated in " << elapsed_time(last_timestamp, get_timestamp()) << "s");

	/*
	// To export the training set
	for(int i = 0; i < training_classes->size(); ++i)
	{
		std::ostringstream output_file;
		output_file << cli_parser.get_export_dir() << "/training_set_" << std::setfill('0') << std::setw(6) << i << ".data";
		fann_save_train(training_sets->operator[](i).get(), output_file.str().c_str());
	}
	*/

	last_timestamp = get_timestamp();
	LOG4CXX_INFO(logger, "Training neural networks");

	boost::shared_ptr< NeuralNetworkVector > networks = train_neural_networks(training_sets);

	LOG4CXX_INFO(logger, "Neural networks trained in " << elapsed_time(last_timestamp, get_timestamp()) << "s");


	tlp::initTulipLib("/home/cyrille/Dev/Tulip/tulip-3.8-svn/debug/install/");
	tlp::loadPlugins(0);


	last_timestamp = get_timestamp();
	LOG4CXX_INFO(logger, "Generating graph structure");

	tlp::DataSet data;
	data.set<int>("Width", haralickImage->GetLargestPossibleRegion().GetSize()[0]);
	data.set<int>("Height", haralickImage->GetLargestPossibleRegion().GetSize()[1]);
	data.set<int>("Depth", haralickImage->GetLargestPossibleRegion().GetSize()[2]);
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

	tlp::BooleanProperty *seed = graph->getLocalProperty<tlp::BooleanProperty>("Seed");
	seed->setAllNodeValue(false);

	{
		tlp::Iterator<tlp::node> *itNodes = graph->getNodes();
		tlp::node u;

		tlp::DoubleVectorProperty *haralick = graph->getLocalProperty<tlp::DoubleVectorProperty>("haralick_feature");

		const double *haralick_features_tmp;
		std::vector<double> haralick_features(8);

		while(itNodes->hasNext())
		{
			u = itNodes->next();
			NormalizedHaralickImage::PixelType texture = haralickImage->GetPixel(haralickImage->ComputeIndex(u.id));

			haralick_features_tmp = texture.GetDataPointer();
			haralick_features.assign(haralick_features_tmp, haralick_features_tmp+8);
			haralick->setNodeValue(u, haralick_features);
		}
		delete itNodes;
	}

	LOG4CXX_INFO(logger, "Graph structure generated in " << elapsed_time(last_timestamp, get_timestamp()) << "s");


	LOG4CXX_INFO(logger, "Classifying pixels with neural networks");

	std::vector< tlp::DoubleProperty* > regularized_segmentations(networks->size()); 

	for(unsigned int i = 0; i < networks->size(); ++i)
	{
		std::ostringstream graph_name;
		graph_name << std::setfill('0') << std::setw(6) << i;

		tlp::Graph* subgraph = graph->addSubGraph(everything, 0, graph_name.str());

		boost::shared_ptr< NeuralNetwork > net = networks->operator[](i);

		tlp::Iterator<tlp::node> *itNodes = subgraph->getNodes();
		tlp::node u;
		tlp::DoubleVectorProperty *f0 = subgraph->getLocalProperty<tlp::DoubleVectorProperty>("f0");
		tlp::DoubleVectorProperty *haralick = subgraph->getProperty<tlp::DoubleVectorProperty>("haralick_feature");
		std::vector<double> features(1);

		while(itNodes->hasNext())
		{
			u = itNodes->next();
			double* result = fann_run( net.get(), const_cast<fann_type *>( &(haralick->getNodeValue(u)[0]) ) ); // Conversion from vector<double> to double*
			features[0] = result[0];
			f0->setNodeValue(u, features);
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

		regularized_segmentations[i] = subgraph->getLocalProperty< DoubleProperty >("fn");

		LOG4CXX_INFO(logger, "Regularization done for image #" << i);
	}

	tlp::saveGraph(graph, cli_parser.get_export_dir() + "/" + "graph.tlp");

	ImageType::Pointer classification_image = ImageType::New();
	classification_image->SetRegions(haralickImage->GetLargestPossibleRegion());
	classification_image->Allocate();
	ImageType::IndexType index;

	int width, height, depth;
	unsigned int id;

	graph->getAttribute<int>("width", width);
	graph->getAttribute<int>("height", height);
	graph->getAttribute<int>("depth", depth);

	tlp::Iterator<tlp::node> *itNodes = graph->getNodes();
	tlp::node u;
	std::vector< double > values(networks->size());

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

		for(unsigned int i = 0; i < networks->size(); ++i)
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

	for(int i = 0; i <= networks->size(); ++i)
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
