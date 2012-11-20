#ifndef COMMON_H
#define COMMON_H

#include "itkImage.h"
#include "itkVectorImage.h"
#include "itkImageFileReader.h"

#define __ImageDimension 3


typedef itk::Image< unsigned char, __ImageDimension > ImageType;

typedef itk::VectorImage< float, __ImageDimension > FeaturesImage;

#endif /* COMMON_H */
