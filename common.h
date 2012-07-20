#ifndef COMMON_H
#define COMMON_H

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkScalarImageToHaralickTextureFeaturesImageFilter.h"
#include "itkComposeImageFilter.h"

#define __ImageDimension 3


typedef itk::Image< unsigned char, __ImageDimension > ImageType;

typedef itk::Image< double, __ImageDimension > ScalarHaralickImageType;
typedef itk::ComposeImageFilter<ScalarHaralickImageType> ScalarHaralickImageToHaralickImageFilterType;
typedef ScalarHaralickImageToHaralickImageFilterType::OutputImageType NormalizedHaralickImage;

#endif /* COMMON_H */
