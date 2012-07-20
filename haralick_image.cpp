#include "common.h"

#include "itkRescaleIntensityImageFilter.h"
#include "itkNthElementImageAdaptor.h"

#include "haralick_image.h"

#include "image_loader.h"

typedef itk::RescaleIntensityImageFilter<ImageType, ImageType> RescaleFilter;
typedef typename itk::Statistics::ScalarImageToHaralickTextureFeaturesImageFilter< ImageType, double > ScalarImageToHaralickTextureFeaturesImageFilter;
typedef typename ScalarImageToHaralickTextureFeaturesImageFilter::OutputImageType HaralickImageType;

typedef itk::NthElementImageAdaptor< HaralickImageType, double > HaralickImageToScalarImageAdaptorType;
typedef itk::RescaleIntensityImageFilter< HaralickImageToScalarImageAdaptorType, ScalarHaralickImageType > ScalarHaralickImageRescaleFilter;

NormalizedHaralickImage::Pointer load_texture_image(const std::string filename, const unsigned int _posterizationLevel, const unsigned int _windowRadius)
{
  // Read the input image
  ImageType::Pointer image;
  try {
    image = ImageLoader::load(filename);
  } catch (ImageLoadingException & ex) {
    throw HaralickImageException(ex.what());
  }

  try {
    // Posterize the input image
    typename RescaleFilter::Pointer rescaler = RescaleFilter::New();
    rescaler->SetInput(image);
    rescaler->SetOutputMinimum(0);
    rescaler->SetOutputMaximum(_posterizationLevel);
    rescaler->Update();

    // Compute the haralick features
    typename ScalarImageToHaralickTextureFeaturesImageFilter::Pointer haralickImageComputer = ScalarImageToHaralickTextureFeaturesImageFilter::New();
    typename ScalarImageToHaralickTextureFeaturesImageFilter::RadiusType windowRadius; windowRadius.Fill(_windowRadius);
    haralickImageComputer->SetInput(rescaler->GetOutput());
    haralickImageComputer->SetWindowRadius(windowRadius);
    haralickImageComputer->SetNumberOfBinsPerAxis(_posterizationLevel);

    typename ScalarImageToHaralickTextureFeaturesImageFilter::OffsetType offset1 = {{0, 1}};
    typename ScalarImageToHaralickTextureFeaturesImageFilter::OffsetVectorType::Pointer offsetV = ScalarImageToHaralickTextureFeaturesImageFilter::OffsetVectorType::New();
    offsetV->push_back(offset1);
    haralickImageComputer->SetOffsets(offsetV);
    haralickImageComputer->Update();

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

    imageToVectorImageFilter->Update();

    return imageToVectorImageFilter->GetOutput();
  }
  catch (itk::ExceptionObject & ex) {
    throw HaralickImageException(ex.what());
  }
}

