#ifndef NEURALNETWORKPIXELCLASSIFIERS_H
#define NEURALNETWORKPIXELCLASSIFIERS_H

#include "doublefann.h"
#include "common.h"

#include <boost/shared_ptr.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <vector>
#include <string>

class TrainingClassException : public std::runtime_error
{
  public:
      TrainingClassException ( const std::string &err ) : std::runtime_error (err) {}
};

class NeuralNetworkPixelClassifiers
{
public:
	typedef struct fann NeuralNetwork;
	typedef std::vector< boost::shared_ptr< NeuralNetwork > > NeuralNetworkVector;

	void init_training_sets(const int number_of_classes);
	void load_training_image(const std::string training_image_filename, const std::vector< std::string > training_classes_filenames);
	void load_training_image(typename FeaturesImage::Pointer training_mage, const std::vector< std::string > training_classes_filenames);

	void build_training_sets();

	void create_and_train_neural_networks( const std::vector< unsigned int > hidden_layers, const float learning_rate, const unsigned int max_epoch, const float mse_target );

	boost::shared_ptr< NeuralNetwork > get_neural_network(const unsigned int i);

	void save_neural_networks(const std::string dir);
	void load_neural_networks(const std::string dir);

	const unsigned int getNumberOfClassifiers() const { return m_NumberOfClassifiers; }
	const unsigned int getNumberOfComponentsPerPixel() const { return m_NumberOfComponentsPerPixel; }

private:
	typedef struct fann_train_data TrainingSet;
	typedef std::vector< boost::shared_ptr< TrainingSet > > TrainingSetVector;

	typedef std::vector< typename FeaturesImage::PixelType > TrainingClass;
	typedef std::vector< boost::shared_ptr< TrainingClass > > TrainingClassVector;

	unsigned int m_NumberOfClasses;
	unsigned int m_NumberOfClassifiers;
	unsigned int m_NumberOfComponentsPerPixel;
	TrainingClassVector m_TrainingClasses;
	TrainingSetVector m_TrainingSets;
	NeuralNetworkVector m_NeuralNetworks;
};

#endif /* NEURALNETWORKPIXELCLASSIFIERS_H */
