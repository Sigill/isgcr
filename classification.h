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

typedef std::vector< typename FeaturesImage::PixelType > TrainingClass;
typedef std::vector< boost::shared_ptr< TrainingClass > > TrainingClassVector;

typedef struct fann_train_data TrainingSet;
typedef std::vector< boost::shared_ptr< TrainingSet > > TrainingSetVector;

typedef struct fann NeuralNetwork;
typedef std::vector< boost::shared_ptr< NeuralNetwork > > NeuralNetworkVector;

boost::shared_ptr<TrainingClassVector> load_classes( const std::vector< std::string > filenames, typename FeaturesImage::Pointer featuresImage );

boost::shared_ptr<TrainingSetVector> generate_training_sets( boost::shared_ptr<TrainingClassVector> raw_learning_classes );

boost::shared_ptr< NeuralNetworkVector > train_neural_networks(boost::shared_ptr<TrainingSetVector> training_sets);

#endif /* CLASSIFICATION_H */
