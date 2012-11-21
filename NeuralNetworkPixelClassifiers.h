#ifndef NEURALNETWORKPIXELCLASSIFIERS_H
#define NEURALNETWORKPIXELCLASSIFIERS_H

#include "doublefann.h"
#include "common.h"

#include <boost/shared_ptr.hpp>
#include <vector>
#include <string>

class LearningClassException : public std::runtime_error
{
  public:
      LearningClassException ( const std::string &err ) : std::runtime_error (err) {}
};

class NeuralNetworkPixelClassifiers
{
public:
	typedef struct fann NeuralNetwork;
	typedef std::vector< boost::shared_ptr< NeuralNetwork > > NeuralNetworkVector;

	void load_training_sets(const std::vector< std::string > filenames, typename FeaturesImage::Pointer featuresImage);
	void train_neural_networks();
	boost::shared_ptr< NeuralNetwork > get_neural_network(const unsigned int i);

private:
	typedef struct fann_train_data TrainingSet;
	typedef std::vector< boost::shared_ptr< TrainingSet > > TrainingSetVector;

	typedef std::vector< typename FeaturesImage::PixelType > TrainingClass;
	typedef std::vector< boost::shared_ptr< TrainingClass > > TrainingClassVector;

	unsigned int m_NumberOfClasses;
	unsigned int m_NumberOfComponentsPerPixel;
	TrainingSetVector m_TrainingSets;
	NeuralNetworkVector m_NeuralNetworks;

};

#endif /* NEURALNETWORKPIXELCLASSIFIERS_H */
