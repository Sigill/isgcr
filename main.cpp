#include <iostream>
#include <vector>

#include "common.h"
#include "time_utils.h"
#include "cli_parser.h"
#include "learning_classes_loader.h"

#include "itkImageFileWriter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkImageRegionIteratorWithIndex.h"

#include "itkNthElementImageAdaptor.h"

#include "doublefann.h"

#include "callgrind.h"

const unsigned int PosterizationLevel = 16;

typedef itk::ImageFileWriter<ImageType> WriterType;
typedef itk::RescaleIntensityImageFilter<ImageType, ImageType> RescaleFilter;

typedef itk::ImageRegionIteratorWithIndex< ImageType > ImageIterator;

typedef itk::NthElementImageAdaptor< HaralickImageType, double > HaralickImageToScalarImageAdaptorType;
typedef itk::RescaleIntensityImageFilter< HaralickImageToScalarImageAdaptorType, ScalarHaralickImageType > ScalarHaralickImageRescaleFilter;


typedef ::itk::Size< __ImageDimension > RadiusType;

int main(int argc, char **argv)
{
  CliParser cli_parser;
  int parse_result = cli_parser.parse_argv(argc, argv);
  if(parse_result <= 0)
    exit(parse_result);

  timestamp_t timestamp_start = get_timestamp();

  // Read the input image
  typename ReaderType::Pointer imageReader = ReaderType::New();
  imageReader->SetFileName(cli_parser.get_input_image());
  imageReader->Update();

  // Posterize the input image
  typename RescaleFilter::Pointer rescaler = RescaleFilter::New();
  rescaler->SetInput(imageReader->GetOutput());
  rescaler->SetOutputMinimum(0);
  rescaler->SetOutputMaximum(PosterizationLevel);
  rescaler->Update();

  // Compute the haralick features
  typename ScalarImageToHaralickTextureFeaturesImageFilter::Pointer haralickImageComputer = ScalarImageToHaralickTextureFeaturesImageFilter::New();
  typename ScalarImageToHaralickTextureFeaturesImageFilter::RadiusType windowRadius; windowRadius.Fill(5);
  haralickImageComputer->SetInput(rescaler->GetOutput());
  haralickImageComputer->SetWindowRadius(windowRadius);
  haralickImageComputer->SetNumberOfBinsPerAxis(PosterizationLevel);

  typename ScalarImageToHaralickTextureFeaturesImageFilter::OffsetType offset1 = {{0, 1}};
  typename ScalarImageToHaralickTextureFeaturesImageFilter::OffsetVectorType::Pointer offsetV = ScalarImageToHaralickTextureFeaturesImageFilter::OffsetVectorType::New();
  offsetV->push_back(offset1);
  haralickImageComputer->SetOffsets(offsetV);
  haralickImageComputer->Update();

  std::cout << "Haralick features computation: " << elapsed_time(timestamp_start, get_timestamp()) << "s" << std::endl;

  // Rescale haralick features
  ScalarHaralickImageToHaralickImageFilterType::Pointer imageToVectorImageFilter = ScalarHaralickImageToHaralickImageFilterType::New();

  #pragma omp parallel for
  for(int i = 0; i < 8; ++i)
  {
    HaralickImageToScalarImageAdaptorType::Pointer adaptor = HaralickImageToScalarImageAdaptorType::New();
    adaptor->SelectNthElement(i);
    adaptor->SetImage(haralickImageComputer->GetOutput());

    ScalarHaralickImageRescaleFilter::Pointer rescaler = ScalarHaralickImageRescaleFilter::New();
    rescaler->SetInput(adaptor);
    rescaler->SetOutputMinimum(0.0);
    rescaler->SetOutputMaximum(1.0);

    rescaler->Update();

    imageToVectorImageFilter->SetInput(i, rescaler->GetOutput());
  }

  std::cout << "Rescaling: " << elapsed_time(timestamp_start, get_timestamp()) << "s" << std::endl;

  imageToVectorImageFilter->Update();

  std::cout << "Combination: " << elapsed_time(timestamp_start, get_timestamp()) << "s" << std::endl;

  typename NormalizedHaralickImage::Pointer haralickImage = imageToVectorImageFilter->GetOutput();
  typename ImageType::RegionType requestedRegion = haralickImage->GetLargestPossibleRegion();

  CALLGRIND_START_INSTRUMENTATION

  boost::shared_ptr< TrainingClassVector > training_classes = load_classes(cli_parser.get_class_images(), haralickImage);
  boost::shared_ptr< TrainingSetVector > training_sets = generate_training_sets(training_classes);

  CALLGRIND_STOP_INSTRUMENTATION

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
