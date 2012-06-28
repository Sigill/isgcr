#include <iostream>
#include <vector>

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkScalarImageToHaralickTextureFeaturesFilter.h"
#include "itkImageRegionIteratorWithIndex.h"

#include "doublefann.h"

const unsigned int D = 2;
const unsigned int PosterizationLevel = 16;

typedef itk::Image<unsigned char, D> ImageType;
typedef itk::ImageFileReader<ImageType> ReaderType;
typedef itk::ImageFileWriter<ImageType> WriterType;
typedef itk::RescaleIntensityImageFilter<ImageType, ImageType> RescaleFilter;

typedef itk::ImageRegionIteratorWithIndex< ImageType > ImageIterator;
typedef typename itk::Statistics::ScalarImageToHaralickTextureFeaturesFilter< ImageType, double  > ScalarImageToHaralickTextureFeaturesFilter;

typedef ::itk::Size< D > RadiusType;

int main(int argc, char **argv)
{
  if(argc != 3)
  {
    std::cout << "Usage: " << argv[0] << " <Image> <LearningMask>" << std::endl << std::endl;
    exit(0);
  }

  typename ReaderType::Pointer imageReader = ReaderType::New();
  imageReader->SetFileName(argv[1]);
  imageReader->Update();

  typename ReaderType::Pointer learningMaskReader = ReaderType::New();
  learningMaskReader->SetFileName(argv[2]);
  learningMaskReader->Update();

  typename RescaleFilter::Pointer rescaler = RescaleFilter::New();
  rescaler->SetInput(imageReader->GetOutput());
  rescaler->SetOutputMinimum(0);
  rescaler->SetOutputMaximum(PosterizationLevel);
  rescaler->Update();

  typename ScalarImageToHaralickTextureFeaturesFilter::Pointer featuresComputer = ScalarImageToHaralickTextureFeaturesFilter::New();
  featuresComputer->SetNumberOfBinsPerAxis(PosterizationLevel);
  featuresComputer->SetInput(rescaler->GetOutput());

  typename ScalarImageToHaralickTextureFeaturesFilter::OffsetType offset1 = {{0, 1}};
  typename ScalarImageToHaralickTextureFeaturesFilter::OffsetVectorType::Pointer offsetV = ScalarImageToHaralickTextureFeaturesFilter::OffsetVectorType::New();
  offsetV->push_back(offset1);
  featuresComputer->SetOffsets(offsetV);

  typename ScalarImageToHaralickTextureFeaturesFilter::InputImageType::RegionType windowRegion;
  typename ScalarImageToHaralickTextureFeaturesFilter::InputImageType::IndexType windowIndex;
  RadiusType windowRadius = {{ 5, 5 }};
  typename ScalarImageToHaralickTextureFeaturesFilter::InputImageType::SizeType windowSize;

  for(unsigned int i = 0; i < D; ++i)
    {
    windowSize.SetElement(i, (windowRadius.GetElement(i) << 1) + 1);
    }

  typename ImageType::RegionType requestedRegion = rescaler->GetOutput()->GetLargestPossibleRegion();

  std::vector< std::vector< double > > raw_inputs;
  std::vector< double > raw_outputs;
  std::vector< double > raw_data(8);

  ImageType::PixelType pix;

  itk::ImageRegionConstIteratorWithIndex< ImageType > imageIterator(rescaler->GetOutput(), requestedRegion);
  itk::ImageRegionConstIteratorWithIndex< ImageType > learningMaskIterator(learningMaskReader->GetOutput(), requestedRegion);
  imageIterator.GoToBegin();
  learningMaskIterator.GoToBegin();
  while(!learningMaskIterator.IsAtEnd())
  {
    pix = learningMaskIterator.Get();
    if(pix == 255)
    {
      raw_outputs.push_back(1.0);
    } else if(pix == 127) {
      raw_outputs.push_back(0.0);
    }

    if(pix > 0)
    {
      windowIndex = imageIterator.GetIndex();
      windowIndex -= windowRadius;

      windowRegion.SetIndex(windowIndex);
      windowRegion.SetSize(windowSize);
      windowRegion.Crop(requestedRegion);

      featuresComputer->SetRegionOfInterest(windowRegion);
      featuresComputer->Update();

      raw_data.clear();
      
      raw_data.push_back( featuresComputer->GetFeature( ScalarImageToHaralickTextureFeaturesFilter::HaralickFeaturesComputer::Energy ) );
      raw_data.push_back( featuresComputer->GetFeature( ScalarImageToHaralickTextureFeaturesFilter::HaralickFeaturesComputer::Entropy ) );
      raw_data.push_back( featuresComputer->GetFeature( ScalarImageToHaralickTextureFeaturesFilter::HaralickFeaturesComputer::Correlation ) );
      raw_data.push_back( featuresComputer->GetFeature( ScalarImageToHaralickTextureFeaturesFilter::HaralickFeaturesComputer::InverseDifferenceMoment ) );
      raw_data.push_back( featuresComputer->GetFeature( ScalarImageToHaralickTextureFeaturesFilter::HaralickFeaturesComputer::Inertia ) );
      raw_data.push_back( featuresComputer->GetFeature( ScalarImageToHaralickTextureFeaturesFilter::HaralickFeaturesComputer::ClusterShade ) );
      raw_data.push_back( featuresComputer->GetFeature( ScalarImageToHaralickTextureFeaturesFilter::HaralickFeaturesComputer::ClusterProminence ) );
      raw_data.push_back( featuresComputer->GetFeature( ScalarImageToHaralickTextureFeaturesFilter::HaralickFeaturesComputer::HaralickCorrelation ) );

      /*
      for(typename std::vector<double>::size_type i=0; i<raw_data.size(); ++i){
        std::cout << raw_data[i] << '\t';
      }
      std::cout << std::endl << std::endl;
      */

      raw_inputs.push_back(raw_data);
    }

    ++learningMaskIterator;
    ++imageIterator;
  }

  struct fann_train_data * training_data = fann_create_train(raw_inputs.size(), 8, 1);

  fann_type **data_input_it = training_data->input;
  fann_type **data_output_it = training_data->output;

  fann_type *data_input_min = new fann_type[8];
  fann_type *data_input_max = new fann_type[8];

  for(unsigned int i = 0; i < 8; ++i)
  {
    data_input_min[i] = 1.0;
    data_input_max[i] = 0.0;
  }

  std::vector< std::vector< double > >::const_iterator raw_inputs_it = raw_inputs.begin(), raw_inputs_end = raw_inputs.end();
  std::vector< double >::const_iterator raw_outputs_it = raw_outputs.begin(), raw_outputs_end = raw_outputs.end();

  while(raw_inputs_it != raw_inputs_end)
  {
    for(unsigned int i = 0; i < 8; ++i)
    {
      (*data_input_it)[i] = (*raw_inputs_it)[i];

      if(data_input_min[i] > (*raw_inputs_it)[i])
        data_input_min[i] = (*raw_inputs_it)[i];

      if(data_input_max[i] < (*raw_inputs_it)[i])
        data_input_max[i] = (*raw_inputs_it)[i];
    }

    ++data_input_it;


    (*data_output_it)[0] = *raw_outputs_it;

    ++data_output_it;


    ++raw_inputs_it;
    ++raw_outputs_it;
  }

  data_input_it = training_data->input;
  for(unsigned int j = 0; j < raw_inputs.size(); ++j)
  {
    for(unsigned int i = 0; i < 8; ++i)
    {
      (*data_input_it)[i] = ((*data_input_it)[i] - data_input_min[i]) / (data_input_max[i] - data_input_min[i]);
    }
    ++data_input_it;
  }

  fann_scale_input_train_data(training_data, 0.0, 1.0);

  fann_save_train(training_data, "training_set.data");

  raw_inputs.clear();
  raw_outputs.clear();

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
    
    input[0] = featuresComputer->GetFeature( ScalarImageToHaralickTextureFeaturesFilter::HaralickFeaturesComputer::Energy );
    input[1] = featuresComputer->GetFeature( ScalarImageToHaralickTextureFeaturesFilter::HaralickFeaturesComputer::Entropy ) ;
    input[2] = featuresComputer->GetFeature( ScalarImageToHaralickTextureFeaturesFilter::HaralickFeaturesComputer::Correlation ) ;
    input[3] = featuresComputer->GetFeature( ScalarImageToHaralickTextureFeaturesFilter::HaralickFeaturesComputer::InverseDifferenceMoment ) ;
    input[4] = featuresComputer->GetFeature( ScalarImageToHaralickTextureFeaturesFilter::HaralickFeaturesComputer::Inertia ) ;
    input[5] = featuresComputer->GetFeature( ScalarImageToHaralickTextureFeaturesFilter::HaralickFeaturesComputer::ClusterShade ) ;
    input[6] = featuresComputer->GetFeature( ScalarImageToHaralickTextureFeaturesFilter::HaralickFeaturesComputer::ClusterProminence ) ;
    input[7] = featuresComputer->GetFeature( ScalarImageToHaralickTextureFeaturesFilter::HaralickFeaturesComputer::HaralickCorrelation );

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

  fann_destroy_train(training_data);

  return 0;
}
