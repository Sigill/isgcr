#ifndef LEARNING_CLASSES_LOADER_H
#define LEARNING_CLASSES_LOADER_H

#include "common.h"
#include "itkImageRegionConstIteratorWithIndex.h"

#include <vector>
#include <boost/shared_ptr.hpp>

#include "doublefann.h"

typedef std::vector< typename NormalizedHaralickImage::PixelType > TrainingClass;
typedef std::vector< boost::shared_ptr< TrainingClass > > TrainingClassVector;

typedef struct fann_train_data TrainingSet;
typedef std::vector< boost::shared_ptr< TrainingSet > > TrainingSetVector;

std::auto_ptr<TrainingSetVector> load_classes(const std::vector< std::string > filenames, NormalizedHaralickImage::Pointer haralickImage);

#endif /* LEARNING_CLASSES_LOADER_H */

