#ifndef NEURALNETWORKPIXELCLASSIFIERS_H
#define NEURALNETWORKPIXELCLASSIFIERS_H

#include "doublefann.h"
#include "common.h"

#include <boost/shared_ptr.hpp>
#include <vector>
#include <string>

#include "ClassificationDataset.h"

class NeuralNetworkPixelClassifiers
{
public:
	typedef struct fann NeuralNetwork;
	typedef std::vector< boost::shared_ptr< NeuralNetwork > > NeuralNetworkVector;

	void create_neural_networks( const int count, const std::vector< unsigned int > layers, const float learning_rate );
	void train_neural_networks(
		boost::shared_ptr< typename ClassificationDataset::FannDatasetVector > training_sets,
		const unsigned int max_epoch,
		const float mse_target,
		boost::shared_ptr< typename ClassificationDataset::FannDatasetVector > validation_sets );

	boost::shared_ptr< NeuralNetwork > get_neural_network(const unsigned int i);

	void save_neural_networks(const std::string dir);
	void load_neural_networks(const std::string dir);

	const unsigned int getNumberOfClassifiers() const { return m_NumberOfClassifiers; }
	const unsigned int getNumberOfComponentsPerPixel() const { return m_NumberOfComponentsPerPixel; }

private:
	unsigned int m_NumberOfClassifiers;
	unsigned int m_NumberOfComponentsPerPixel;
	NeuralNetworkVector m_NeuralNetworks;
};

#endif /* NEURALNETWORKPIXELCLASSIFIERS_H */
