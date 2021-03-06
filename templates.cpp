#ifndef TEMPLATES_CPP
#define TEMPLATES_CPP

#undef ITK_MANUAL_INSTANTIATION
#undef MANUAL_INSTANTIATION

#include "common.h"

#include <itkImageSeriesReader.h>
#include <itkImageSeriesWriter.h>

#include <itkBinaryThresholdImageFilter.h>
#include <itkImageRegionConstIteratorWithIndex.h>

#include <doublefann.h>
#include "ClassificationDataset.h"

#include "Classifier.h"

#include <itkSimpleDataObjectDecorator.h>
#include <string>

template class itk::SimpleDataObjectDecorator<std::string>;

template class itk::VariableLengthVector<float>;

template class itk::Image< unsigned char, __ImageDimension >;
template class itk::VectorImage< float, __ImageDimension >;

template class itk::ImageSource< ImageType >;
template class itk::ImageSource< FeaturesImage >;

template class itk::ImageFileReader< FeaturesImage >;
template class itk::ImageSeriesReader< ImageType >;

template class itk::Image< unsigned char, 2 >;
template class itk::ImageSeriesWriter< ImageType, itk::Image< unsigned char, 2 > >;

template class itk::BinaryThresholdImageFilter< ImageType, ImageType >;

template class itk::ImageRegionConstIteratorWithIndex< ImageType >;

template class ClassificationDataset<fann_type>;

template class Classifier<fann_type>;

#endif /* TEMPLATES_CPP */
