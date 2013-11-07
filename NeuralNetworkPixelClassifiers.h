#ifndef NEURALNETWORKPIXELCLASSIFIERS_H
#define NEURALNETWORKPIXELCLASSIFIERS_H

#include "doublefann.h"
#include "common.h"

#include <boost/shared_ptr.hpp>
#include <vector>
#include <string>

#include "Classifier.h"

#include "FannClassificationDataset.h"

// Forward declaration
namespace std {
	template <class T1, class T2> struct pair;
}

class NeuralNetworkPixelClassifiers : public Classifier< fann_type >
{
public:
	void create_neural_networks( const int count, const std::vector< unsigned int > layers, const float learning_rate );
	void train_neural_networks(
		FannClassificationDataset const *training_sets,
		const unsigned int max_epoch,
		const float mse_target,
		FannClassificationDataset const *validation_sets );

	void save(const std::string dir);
	void load(const std::string dir);
	std::vector<float> classify(const std::vector< InputValueType > &input) const;

	const unsigned int getNumberOfClassifiers() const { return m_NumberOfClassifiers; }
	const unsigned int getNumberOfComponentsPerPixel() const { return m_NumberOfComponentsPerPixel; }

private:
	typedef struct fann NeuralNetwork;
	typedef std::vector< boost::shared_ptr< NeuralNetwork > > NeuralNetworkVector;

	unsigned int m_NumberOfClassifiers;
	unsigned int m_NumberOfComponentsPerPixel;
	NeuralNetworkVector m_NeuralNetworks;
	std::vector< std::vector< std::pair< float, float > > > m_TrainingScoresHistory;
};

#endif /* NEURALNETWORKPIXELCLASSIFIERS_H */
