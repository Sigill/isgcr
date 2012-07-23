#ifndef CLASSIFICATION_H
#define CLASSIFICATION_H

#include "common.h"
#include "itkImageRegionConstIteratorWithIndex.h"

#include <stdexcept>
#include <vector>
#include <boost/shared_ptr.hpp>

#include "doublefann.h"

class LearningClassException : public std::runtime_error
{
  public:
      LearningClassException ( const std::string &err ) : std::runtime_error (err) {}
};

typedef std::vector< typename NormalizedHaralickImage::PixelType > TrainingClass;
typedef std::vector< boost::shared_ptr< TrainingClass > > TrainingClassVector;

typedef struct fann_train_data TrainingSet;
typedef std::vector< boost::shared_ptr< TrainingSet > > TrainingSetVector;


boost::shared_ptr<TrainingClassVector> load_classes( const std::vector< std::string > filenames, NormalizedHaralickImage::Pointer haralickImage ) throw(LearningClassException);

boost::shared_ptr<TrainingSetVector> generate_training_sets( boost::shared_ptr<TrainingClassVector> raw_learning_classes );


#endif /* CLASSIFICATION_H */
