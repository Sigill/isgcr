#include <iostream>
#include <vector>

#include "common.h"
#include "time_utils.h"
#include "cli_parser.h"
#include "classification.h"
#include "image_loader.h"

#include "itkImageFileWriter.h"
#include "itkImageRegionIteratorWithIndex.h"

#include "haralick.h"

#include "doublefann.h"

#include <tulip/Graph.h>
#include <tulip/TlpTools.h>
#include <tulip/TulipPlugin.h>

#include <boost/filesystem.hpp>

#include "callgrind.h"

using namespace tlp;
using namespace std;

const unsigned int posterizationLevel = 16;
const unsigned int windowRadius = 5;

typedef itk::ImageFileWriter<ImageType> WriterType;

typedef itk::ImageRegionIteratorWithIndex< ImageType > ImageIterator;

typedef ::itk::Size< __ImageDimension > RadiusType;

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

	NormalizedHaralickImage::Pointer haralickImage = load_texture_image(cli_parser.get_input_image(), posterizationLevel, windowRadius);

	std::cout << "Computation of Haralick features: " << elapsed_time(timestamp_start, get_timestamp()) << "s" << std::endl;

	boost::shared_ptr< TrainingClassVector > training_classes;
	try {
		training_classes = load_classes(cli_parser.get_class_images(), haralickImage);
	} catch (LearningClassException & ex) {
		std::cerr << "Unable to load the training classes: " << ex.what() << std::endl;
	}

	boost::shared_ptr< TrainingSetVector > training_sets = generate_training_sets(training_classes);

	for(int i = 0; i < training_classes->size(); ++i)
	{
		std::ostringstream output_file;
		output_file << cli_parser.get_export_dir() << "/training_set_" << std::setfill('0') << std::setw(6) << i << ".data";
		fann_save_train(training_sets->operator[](i).get(), output_file.str().c_str());
	}

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

	for(unsigned int i = 0; i < networks->size(); ++i)
	{
		std::cout << "Applying CV_Ta algorithm on image #" << i << std::endl;
		tlp::Graph *graph = tlp::importGraph("Grid 3D", data);

		std::cout << "Grid created for image #" << i << std::endl;

		tlp::DoubleProperty *weight = graph->getLocalProperty<tlp::DoubleProperty>("Weight");
		weight->setAllEdgeValue(1);

		tlp::BooleanProperty *roi = graph->getLocalProperty<tlp::BooleanProperty>("Roi");
		roi->setAllNodeValue(true);

		tlp::BooleanProperty *seed = graph->getLocalProperty<tlp::BooleanProperty>("Seed");
		roi->setAllNodeValue(true);

		boost::shared_ptr< NeuralNetwork > net = networks->operator[](i);

		tlp::Iterator<tlp::node> *itNodes = graph->getNodes();
		tlp::node u;
		tlp::DoubleVectorProperty *f0 = graph->getLocalProperty<tlp::DoubleVectorProperty>("f0");
		tlp::DoubleVectorProperty *haralick = graph->getLocalProperty<tlp::DoubleVectorProperty>("haralick_feature");
		std::vector<double> features(1);
		const double *haralick_features_tmp;
		std::vector<double> haralick_features(8);
		while(itNodes->hasNext())
		{
			u = itNodes->next();
			NormalizedHaralickImage::PixelType texture = haralickImage->GetPixel(haralickImage->ComputeIndex(u.id));
			double* result = fann_run(net.get(), const_cast<fann_type *>(texture.GetDataPointer()));
			features[0] = result[0];
			f0->setNodeValue(u, features);

			haralick_features_tmp = texture.GetDataPointer();
			haralick_features.assign(haralick_features_tmp, haralick_features_tmp+8);
			haralick->setNodeValue(u, haralick_features);
		}
		delete itNodes;

		std::cout << "Data copied for image #" << i << std::endl;

		std::ostringstream output_graph;
		output_graph << cli_parser.get_export_dir() << "/graph_" << std::setfill('0') << std::setw(6) << i << ".tlp";
		tlp::saveGraph(graph, output_graph.str());

		std::ostringstream output_dir;
		output_dir << cli_parser.get_export_dir() << "/" << std::setfill('0') << std::setw(6) << i;

		DataSet data4;
		data4.set<PropertyInterface*>("Data", f0);
		data4.set<PropertyInterface*>("Mask", graph->getLocalProperty<BooleanProperty>("Seed"));
		data4.set<unsigned int>("Number of iterations", 100);
		data4.set<double>("Lambda1", 0.25);
		data4.set<double>("Lambda2", 0.25);
		data4.set<unsigned int>("Export interval", 50);
		data4.set<string>("dir::Export directory", output_dir.str());
		data4.set<PropertyInterface*>("Weight", graph->getLocalProperty<DoubleProperty>("Weight"));
		data4.set<PropertyInterface*>("Roi", graph->getLocalProperty<BooleanProperty>("Roi"));

		string error4;
		if(!graph->applyAlgorithm("Cv_Ta", error4, &data4)) {
			std::cerr << "Unable to apply the Cv_Ta algorithm: " << error4 << std::endl;
			return -1;
		}

		std::cout << "Regularization done for image #" << i << std::endl;

		delete graph;
	}

	/*
	   fann_shuffle_train_data(training_data);

	   struct fann* ann = fann_create_standard(3, 8, 3, 1);
	   fann_set_activation_function_hidden(ann, FANN_SIGMOID);
	   fann_set_activation_function_output(ann, FANN_SIGMOID);
	   fann_set_train_stop_function(ann, FANN_STOPFUNC_MSE);
	   fann_set_learning_rate(ann, 0.1);

	   fann_train_on_data(ann, training_data, 1000, 0, 0.0001);
	   std::cout << fann_get_MSE(ann) << std::endl;



	   ImageType::Pointer out = ImageType::New();
	   out->SetRegions(requestedRegion);
	   out->Allocate();

	   ImageIterator in_it(rescaler->GetOutput(), rescaler->GetOutput()->GetLargestPossibleRegion());
	   ImageIterator out_it(out, rescaler->GetOutput()->GetLargestPossibleRegion());

	   fann_type *input = new fann_type[8];
	   fann_type *output;

	   while(!in_it.IsAtEnd())
	   {
	   windowIndex = in_it.GetIndex();
	   windowIndex -= windowRadius;

	   windowRegion.SetIndex(windowIndex);
	   windowRegion.SetSize(windowSize);
	   windowRegion.Crop(requestedRegion);

	   featuresComputer->SetRegionOfInterest(windowRegion);
	   featuresComputer->Update();

	   raw_data.clear();

	   input[0] = featuresComputer->GetFeature( ScalarImageToLocalHaralickTextureFeaturesFilter::HaralickFeaturesComputer::Energy );
	   input[1] = featuresComputer->GetFeature( ScalarImageToLocalHaralickTextureFeaturesFilter::HaralickFeaturesComputer::Entropy ) ;
	   input[2] = featuresComputer->GetFeature( ScalarImageToLocalHaralickTextureFeaturesFilter::HaralickFeaturesComputer::Correlation ) ;
	   input[3] = featuresComputer->GetFeature( ScalarImageToLocalHaralickTextureFeaturesFilter::HaralickFeaturesComputer::InverseDifferenceMoment ) ;
	   input[4] = featuresComputer->GetFeature( ScalarImageToLocalHaralickTextureFeaturesFilter::HaralickFeaturesComputer::Inertia ) ;
	   input[5] = featuresComputer->GetFeature( ScalarImageToLocalHaralickTextureFeaturesFilter::HaralickFeaturesComputer::ClusterShade ) ;
	   input[6] = featuresComputer->GetFeature( ScalarImageToLocalHaralickTextureFeaturesFilter::HaralickFeaturesComputer::ClusterProminence ) ;
	   input[7] = featuresComputer->GetFeature( ScalarImageToLocalHaralickTextureFeaturesFilter::HaralickFeaturesComputer::HaralickCorrelation );

	   for(unsigned int i = 0; i < 8; ++i)
	   {
	   input[i] = (input[i] - data_input_min[i]) / (data_input_max[i] - data_input_min[i]);
	   }

	   output = fann_run(ann, input);

	   out_it.Set((unsigned int)(output[0] * 255));

	   ++in_it;
	   ++out_it;
	   }

	   WriterType::Pointer writer = WriterType::New();
	   writer->SetInput(out);
	   writer->SetFileName("out.bmp");
	   writer->Update();
	 */

	return 0;
}
