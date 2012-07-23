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

#include "callgrind.h"

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

  timestamp_t timestamp_start = get_timestamp();

  typename NormalizedHaralickImage::Pointer haralickImage = load_texture_image(cli_parser.get_input_image(), posterizationLevel, windowRadius);

  std::cout << "Computation of Haralick features: " << elapsed_time(timestamp_start, get_timestamp()) << "s" << std::endl;

  boost::shared_ptr< TrainingClassVector > training_classes;
  try {
    training_classes = load_classes(cli_parser.get_class_images(), haralickImage);
  } catch (LearningClassException & ex) {
    std::cerr << "Unable to load the training classes: " << ex.what() << std::endl;
  }

  boost::shared_ptr< TrainingSetVector > training_sets = generate_training_sets(training_classes);

  const unsigned int number_of_classes = training_sets->size();

  #pragma omp parallel for if(number_of_classes > 1)
  for(int i = 0; i < number_of_classes; ++i)
  {
    std::cout << "Training ann #" << i << std::endl;
    boost::shared_ptr<TrainingSet> current_training_set = training_sets->operator[](i);

    fann_shuffle_train_data(current_training_set.get());

    struct fann* ann = fann_create_standard(3, 8, 3, 1);
    fann_set_activation_function_hidden(ann, FANN_SIGMOID);
    fann_set_activation_function_output(ann, FANN_SIGMOID);
    fann_set_train_stop_function(ann, FANN_STOPFUNC_MSE);
    fann_set_learning_rate(ann, 0.1);

    fann_train_on_data(ann, current_training_set.get(), 1000, 0, 0.0001);
    std::cout << "MSE for ann #" << i << ": " << fann_get_MSE(ann) << std::endl;
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
