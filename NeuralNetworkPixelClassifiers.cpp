#include "NeuralNetworkPixelClassifiers.h"
#include "image_loader.h"

#include "itkImageRegionConstIteratorWithIndex.h"

#include "log4cxx/logger.h"

#include <boost/filesystem.hpp>
#include <boost/regex.hpp>

#include <string>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <algorithm>

struct _StringComparator {
	  bool operator() (const std::string a, const std::string b) { return a.compare(b);}
} StringComparator;

void NeuralNetworkPixelClassifiers::create_neural_networks( const int count, const std::vector< unsigned int > layers, const float learning_rate )
{
	log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("main"));

	m_NumberOfClassifiers = count;
	m_NumberOfComponentsPerPixel = layers[0];

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

void NeuralNetworkPixelClassifiers::train_neural_networks( boost::shared_ptr< typename ClassificationDataset::FannDatasetVector > training_sets, const unsigned int max_epoch, const float mse_target )
{
	log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("main"));

	#pragma omp parallel for
	for(int i = 0; i < m_NumberOfClassifiers; ++i)
	{
		LOG4CXX_INFO(logger, "Training ann #" << i);

		boost::shared_ptr< NeuralNetwork > current_neural_network = m_NeuralNetworks[i];
		boost::shared_ptr< ClassificationDataset::FannDataset> current_training_set = training_sets->operator[](i);

		fann_shuffle_train_data(current_training_set.get());

		fann_train_on_data(current_neural_network.get(), current_training_set.get(), max_epoch, 0, mse_target);

		LOG4CXX_INFO(logger, "MSE for ann #" << i << ": " << fann_get_MSE(current_neural_network.get()));
	}
}

boost::shared_ptr< typename NeuralNetworkPixelClassifiers::NeuralNetwork >
NeuralNetworkPixelClassifiers
::get_neural_network(const unsigned int i)
{
	return m_NeuralNetworks[i];
}

void NeuralNetworkPixelClassifiers::save_neural_networks(const std::string dir)
{
	log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("main"));
	LOG4CXX_INFO(logger, "Saving neural networks in " << dir);

	for(int i = 0; i < m_NumberOfClassifiers; ++i) {
		std::ostringstream filename;
		filename << std::setfill('0') << std::setw(6) << (i+1) << ".ann";

		boost::filesystem::path path = boost::filesystem::path(dir) / filename.str();

		if(0 != fann_save(m_NeuralNetworks[i].get(), path.native().c_str())) {
			throw std::runtime_error("Cannot save neural network in " + path.native());
		}
	}
}

void NeuralNetworkPixelClassifiers::load_neural_networks(const std::string dir)
{
	log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("main"));
	LOG4CXX_INFO(logger, "Loading neural networks from " << dir);

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

	for(std::vector<std::string>::const_iterator it = config_files.begin(); it != config_files.end(); ++it) {
		LOG4CXX_INFO(logger, "Loading neural network from " << *it);

		NeuralNetwork* ann = fann_create_from_file(it->c_str());

		if(ann == NULL)
			throw std::runtime_error("Cannot load neural network from " + *it);

		m_NeuralNetworks.push_back( boost::shared_ptr< NeuralNetwork >( ann, fann_destroy ) );
	}

	m_NumberOfComponentsPerPixel = fann_get_num_input(m_NeuralNetworks.front().get());

	LOG4CXX_INFO(logger, "Number of components per pixel: " << m_NumberOfComponentsPerPixel);
}
