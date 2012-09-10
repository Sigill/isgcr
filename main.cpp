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

#include "callgrind.h"

using namespace tlp;
using namespace std;

int main(int argc, char **argv)
{
	CliParser cli_parser;
	int parse_result = cli_parser.parse_argv(argc, argv);
	if(parse_result <= 0)
		exit(parse_result);

	// Creation of the export folders for each class
	for(int i = 0; i < cli_parser.get_class_images().size(); ++i)
	{
		std::ostringstream output_dir;
		output_dir << cli_parser.get_export_dir() << "/" << std::setfill('0') << std::setw(6) << i;

		boost::filesystem::path path_output_dir(output_dir.str());

		if(boost::filesystem::exists(path_output_dir)) {
			if(boost::filesystem::is_directory(path_output_dir)) {
				if(!boost::filesystem::is_empty(path_output_dir)) {
					std::cerr << "Output dir (" << path_output_dir.string() << ") exists but is not empty" << std::endl;
					return -1;
				}
			} else {
				std::cerr << "Output dir (" << path_output_dir.string() << ") already exists as a file" << std::endl;
				return -1;
			}
		} else {
			if(!boost::filesystem::create_directories(path_output_dir)) {
				std::cerr << "Output dir (" << path_output_dir.string() << ") cannot be created" << std::endl;
				return -1;
			}
		}
	}

	timestamp_t timestamp_start = get_timestamp();

	NormalizedHaralickImage::Pointer haralickImage = load_texture_image(cli_parser.get_input_image(), cli_parser.get_num_gray(), cli_parser.get_window_radius(), cli_parser.get_offset());

	std::cout << "Computation of Haralick features: " << elapsed_time(timestamp_start, get_timestamp()) << "s" << std::endl;

	boost::shared_ptr< TrainingClassVector > training_classes;
	try {
		training_classes = load_classes(cli_parser.get_class_images(), haralickImage);
	} catch (LearningClassException & ex) {
		std::cerr << "Unable to load the training classes: " << ex.what() << std::endl;
	}

	boost::shared_ptr< TrainingSetVector > training_sets = generate_training_sets(training_classes);

	/*
	// To export the training set
	for(int i = 0; i < training_classes->size(); ++i)
	{
		std::ostringstream output_file;
		output_file << cli_parser.get_export_dir() << "/training_set_" << std::setfill('0') << std::setw(6) << i << ".data";
		fann_save_train(training_sets->operator[](i).get(), output_file.str().c_str());
	}
	*/

	boost::shared_ptr< NeuralNetworkVector > networks = train_neural_networks(training_sets);

	tlp::initTulipLib("/home/cyrille/Dev/Tulip/tulip-3.8-svn/debug/install/");
	tlp::loadPlugins(0);

	tlp::DataSet data;
	data.set<int>("Width", haralickImage->GetLargestPossibleRegion().GetSize()[0]);
	data.set<int>("Height", haralickImage->GetLargestPossibleRegion().GetSize()[1]);
	data.set<int>("Depth", haralickImage->GetLargestPossibleRegion().GetSize()[2]);
	data.set<tlp::StringCollection>("Connectivity", tlp::StringCollection("4"));
	data.set<bool>("Positionning", true);
	data.set<double>("Spacing", 1.0);


	tlp::Graph *graph = tlp::importGraph("Grid 3D", data);

	std::cout << "Graph structure created" << std::endl;

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

	std::cout << "Haralick features copied" << std::endl;

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
			double* result = fann_run( net.get(), const_cast<fann_type *>( &(haralick->getNodeValue(u)[0]) ) );
			features[0] = result[0];
			f0->setNodeValue(u, features);
		}
		delete itNodes;

		std::cout << "Data classification done for image #" << i << std::endl;

		std::cout << "Applying CV_Ta algorithm on image #" << i << std::endl;

		{
			std::ostringstream output_graph;
			output_graph << cli_parser.get_export_dir() << "/graph_" << std::setfill('0') << std::setw(6) << i << ".tlp";
			tlp::saveGraph(subgraph, output_graph.str());
		}

		std::ostringstream output_dir;
		output_dir << cli_parser.get_export_dir() << "/" << std::setfill('0') << std::setw(6) << i;

		DataSet data4;
		data4.set<PropertyInterface*>("Data", f0);
		data4.set<PropertyInterface*>("Mask", seed);
		data4.set<unsigned int>("Number of iterations", cli_parser.get_num_iter());
		data4.set<double>("Lambda1", cli_parser.get_lambda1());
		data4.set<double>("Lambda2", cli_parser.get_lambda2());
		data4.set<unsigned int>("Export interval", cli_parser.get_export_interval());
		data4.set<string>("dir::Export directory", output_dir.str());
		data4.set<PropertyInterface*>("Weight", weight);
		data4.set<PropertyInterface*>("Roi", roi);

		std::cout << "Applying the Cv_Ta algorithm on image #" << i << std::endl;
		string error4;
		if(!subgraph->applyAlgorithm("Cv_Ta", error4, &data4)) {
			std::cerr << "Unable to apply the Cv_Ta algorithm: " << error4 << std::endl;
			return -1;
		}

		regularized_segmentations[i] = subgraph->getLocalProperty< DoubleProperty >("fn");

		std::cout << "Regularization done for image #" << i << std::endl;
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
			max_pos = networks->size();
		else
			max_pos = std::distance(values.begin(), max_it);

		classification_image->SetPixel(index, max_pos);

		//copy( &values[0], &values[networks->size()], std::ostream_iterator< double >(std::cout, ", "));
		//std::cout << std::endl;
	}
	delete itNodes;

	std::string final_output = cli_parser.get_export_dir() + "/final";
	if(!boost::filesystem::create_directories(boost::filesystem::path(final_output)))
	{
		std::cerr << final_output << " cannot be created" << std::endl;
		return -1;
	}

	itk::NumericSeriesFileNames::Pointer outputNames = itk::NumericSeriesFileNames::New();
	final_output = final_output + "/%06d.bmp";
	outputNames->SetSeriesFormat(final_output.c_str());
	outputNames->SetStartIndex(1);
	outputNames->SetEndIndex(depth);


	typedef itk::ImageSeriesWriter< ImageType, itk::Image< unsigned char, 2 > > WriterType;
	WriterType::Pointer writer = WriterType::New();
	writer->SetInput(classification_image);
	//writer->SetSeriesFormat(final_output.c_str());
	writer->SetFileNames(outputNames->GetFileNames());
	writer->Update();

	delete graph;

	return 0;
}
