#include "NeuralNetworkPixelClassifiers.h"
#include "image_loader.h"

#include "itkImageRegionConstIteratorWithIndex.h"

#include "log4cxx/logger.h"

#include <boost/filesystem.hpp>
#include <boost/regex.hpp>

#include <string>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <utility>

struct _StringComparator {
	  bool operator() (const std::string a, const std::string b) { return a < b;}
} StringComparator;

void NeuralNetworkPixelClassifiers::create_neural_networks( const unsigned int inputSize, const unsigned int numberOfClassifiers, const std::vector< unsigned int > hiddenLayers, const float learning_rate )
{
	log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("main"));

	m_InputSize = inputSize;
	m_NumberOfClassifiers = numberOfClassifiers;
	m_NumberOfClasses = 1 == m_NumberOfClassifiers ? 2 : m_NumberOfClassifiers;

	std::vector< unsigned int > layers = hiddenLayers;
	layers.insert(layers.begin(), m_InputSize);
	layers.push_back(1);

	for(int i = 0; i < m_NumberOfClassifiers; ++i)
	{
		NeuralNetwork* ann = fann_create_standard_array(layers.size(), layers.data());
		fann_set_activation_function_hidden(ann, FANN_SIGMOID);
		fann_set_activation_function_output(ann, FANN_SIGMOID);
		fann_set_train_stop_function(ann, FANN_STOPFUNC_MSE);
		fann_set_learning_rate(ann, learning_rate);

		m_NeuralNetworks.push_back( boost::shared_ptr< NeuralNetwork >( ann, fann_destroy ) );
	}
}

bool score_comparator(std::pair< float, float> & a, std::pair< float, float> & b)
{
	return a.second < b.second;
}

void NeuralNetworkPixelClassifiers::train_neural_networks(
	FannClassificationDataset const *training_sets,
	const unsigned int max_epoch,
	const float mse_target,
	FannClassificationDataset const *validation_sets )
{
	log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("main"));

	m_TrainingScoresHistory.clear();

	if(validation_sets != NULL) {
		m_TrainingScoresHistory.reserve(m_NumberOfClassifiers);
		for(int i = 0; i < m_NumberOfClassifiers; ++i)
			m_TrainingScoresHistory.push_back(std::vector< std::pair< float, float > >());
	}

	#pragma omp parallel for
	for(int i = 0; i < m_NumberOfClassifiers; ++i)
	{
		LOG4CXX_INFO(logger, "Training ann #" << i);

		boost::shared_ptr< NeuralNetwork > current_neural_network = m_NeuralNetworks[i];
		FannClassificationDataset::FannDataset *current_training_set = training_sets->getSet(i);

		if(validation_sets == NULL)
		{
			fann_train_on_data(current_neural_network.get(), current_training_set, max_epoch, 0, mse_target);
		} else {
			FannClassificationDataset::FannDataset *current_validation_set = validation_sets->getSet(i);

			std::vector< boost::shared_ptr< NeuralNetwork > > trainingHistory;
			trainingHistory.reserve(max_epoch);

			for(int j = 0; j < max_epoch; ++j) {
				float train_mse      = fann_train_epoch( current_neural_network.get(), current_training_set  ),
				      validation_mse = fann_test_data(   current_neural_network.get(), current_validation_set );

				trainingHistory.push_back( boost::shared_ptr< NeuralNetwork >( fann_copy(current_neural_network.get()), fann_destroy ) );

				m_TrainingScoresHistory[i].push_back(std::make_pair(train_mse, validation_mse));

				LOG4CXX_INFO(logger, "MSE for ann #" << i << "; " << j << "; " << train_mse << "; " << validation_mse);
			}

			const int best_neural_network_index = std::distance(
					m_TrainingScoresHistory[i].begin(),
					std::min_element(m_TrainingScoresHistory[i].begin(), m_TrainingScoresHistory[i].end(), score_comparator)
				);

			LOG4CXX_INFO(logger, "Best neural network for dataset #" << i << " obtained at training-iteration #" << best_neural_network_index << ": MSE=" << m_TrainingScoresHistory[i][best_neural_network_index].second);

			m_NeuralNetworks[i] = boost::shared_ptr< NeuralNetwork >(trainingHistory[best_neural_network_index]);

			fann_test_data(m_NeuralNetworks[i].get(), current_validation_set);
		}

		LOG4CXX_INFO(logger, "MSE for ann #" << i << ": " << fann_get_MSE(m_NeuralNetworks[i].get()));
	}
}

void NeuralNetworkPixelClassifiers::save(const std::string dir)
{
	log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("main"));
	LOG4CXX_INFO(logger, "Saving neural networks in " << dir);

	for(int i = 0; i < m_NumberOfClassifiers; ++i) {
		std::ostringstream filename;
		filename << std::setfill('0') << std::setw(6) << (i+1) << ".ann";

		boost::filesystem::path path = boost::filesystem::path(dir) / filename.str();

		if(0 != fann_save(m_NeuralNetworks[i].get(), path.native().c_str())) {
			throw std::runtime_error("Cannot save the neural-network in " + path.native());
		}

		if(!m_TrainingScoresHistory.empty()) {
			std::ostringstream filename;
			filename << std::setfill('0') << std::setw(6) << (i+1) << "-training-scores.dat";

			boost::filesystem::path path = boost::filesystem::path(dir) / filename.str();

			std::ofstream score_file;
			score_file.exceptions(std::ofstream::failbit | std::ofstream::badbit);
			score_file.open(path.native().c_str(), std::ios::out | std::ios::trunc); // This will erase the content of the file

			try {
				for(std::vector< std::pair< float, float > >::const_iterator it = m_TrainingScoresHistory[i].begin(); it != m_TrainingScoresHistory[i].end(); ++it) {
					score_file << it->first << "\t" << it->second << std::endl;
				}

				score_file.close();
			} catch(std::ifstream::failure &e) {
				throw std::runtime_error("Cannot save the neural-network training-scores file in " + path.native() + " (" + e.what() + ")");
			}
		}
	}
}

void NeuralNetworkPixelClassifiers::load(const std::string dir)
{
	log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("main"));
	LOG4CXX_INFO(logger, "Loading neural networks from " << dir);

	m_TrainingScoresHistory.clear();

	const boost::regex config_file_filter( "\\d{6,6}.ann" );
	std::vector< std::string > config_files;

	boost::filesystem::directory_iterator end_itr; // Default ctor yields past-the-end
	for( boost::filesystem::directory_iterator i( dir ); i != end_itr; ++i ) {
		// Skip if not a file
		if( !boost::filesystem::is_regular_file( i->status() ) ) continue;

		boost::smatch what;

		// Skip if no match
		if( !boost::regex_match( i->path().filename().native(), what, config_file_filter ) ) continue;

		// File matches, store it
		config_files.push_back( i->path().native() );
	}

	std::sort(config_files.begin(), config_files.end(), StringComparator);

	m_NumberOfClassifiers = config_files.size();
	m_NumberOfClasses = 1 == m_NumberOfClassifiers ? 2 : m_NumberOfClassifiers;

	for(std::vector<std::string>::const_iterator it = config_files.begin(); it != config_files.end(); ++it) {
		LOG4CXX_INFO(logger, "Loading neural network from " << *it);

		NeuralNetwork* ann = fann_create_from_file(it->c_str());

		if(ann == NULL)
			throw std::runtime_error("Cannot load neural network from " + *it);

		m_NeuralNetworks.push_back( boost::shared_ptr< NeuralNetwork >( ann, fann_destroy ) );
	}

	m_InputSize = fann_get_num_input(m_NeuralNetworks.front().get());

	LOG4CXX_INFO(logger, "Number of components per pixel: " << m_InputSize);
}

std::vector<float> NeuralNetworkPixelClassifiers::classify(const std::vector< fann_type > &input) const
{
	std::vector<float> result(m_NumberOfClassifiers);

	for(int i = 0; i < m_NumberOfClassifiers; ++i)
	{
		double* r = fann_run( m_NeuralNetworks[i].get(), const_cast<fann_type *>( input.data() ) );
		result[i] = r[0];
	}

	return result;
}
