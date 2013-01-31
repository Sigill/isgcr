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

void NeuralNetworkPixelClassifiers::init_training_sets(const int number_of_classes)
{
	this->m_NumberOfClasses = number_of_classes;

	this->m_NumberOfComponentsPerPixel = 0;

	//this->m_TrainingSets = new TrainingClassVector();
	this->m_TrainingClasses = TrainingClassVector(this->m_NumberOfClasses);
	for(int i = 0; i < this->m_NumberOfClasses; ++i) {
		this->m_TrainingClasses[i] = boost::shared_ptr< TrainingClass >(new TrainingClass());
	}

	this->m_NumberOfClassifiers = (this->m_NumberOfClasses == 2 ? 1 : this->m_NumberOfClasses);
}

void NeuralNetworkPixelClassifiers::load_training_image(const std::string training_image_filename, const std::vector< std::string > training_classes_filenames)
{
	typename itk::ImageFileReader< FeaturesImage >::Pointer reader = itk::ImageFileReader< FeaturesImage >::New();
	reader->SetFileName(training_image_filename);

	try {
		reader->Update();
	} catch( itk::ExceptionObject &ex ) {
		std::stringstream err;
		err << "ITK is unable to load the image \"" << training_image_filename << "\" (" << ex.what() << ")";

		throw TrainingClassException(err.str());
	}

	this->load_training_image(reader->GetOutput(), training_classes_filenames);
}

void NeuralNetworkPixelClassifiers::load_training_image(typename FeaturesImage::Pointer training_image, const std::vector< std::string > training_classes_filenames)
{
	log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("main"));

	if(this->m_NumberOfComponentsPerPixel == 0)
		this->m_NumberOfComponentsPerPixel = training_image->GetNumberOfComponentsPerPixel();
	else if(this->m_NumberOfComponentsPerPixel != training_image->GetNumberOfComponentsPerPixel())
		throw TrainingClassException("The image has a number of components which is unexpected.");

	/**
	  * Loading the classes.
	  * For each learning class we build a vector of the pixels
	  * to be used during learning.
	  */
	for(int i = 0; i < m_NumberOfClasses; ++i)
	{
		LOG4CXX_INFO(logger, "Loading class from " << training_classes_filenames[i]);

		ImageType::Pointer image;
		try {
			image = ImageLoader::load(training_classes_filenames[i]);
		} catch (ImageLoadingException & ex) {
			throw TrainingClassException(ex.what());
		} 

		if(image->GetLargestPossibleRegion().GetSize() != training_image->GetLargestPossibleRegion().GetSize()) {
			std::stringstream err;
			err << "The dimensions of the training class image \"" << training_classes_filenames[i] << "\" (" << image->GetLargestPossibleRegion().GetSize()
				<< ") differs from the dimensions of the training image (" << training_image->GetLargestPossibleRegion().GetSize() << ")";

			throw TrainingClassException(err.str());
		}

		TrainingClass* current_class = this->m_TrainingClasses[i].get();

		typename itk::ImageRegionConstIteratorWithIndex< ImageType > learningClassIterator(image, image->GetLargestPossibleRegion());
		while(!learningClassIterator.IsAtEnd())
		{
			if(255 == learningClassIterator.Get()) {
				current_class->push_back(training_image->GetPixel(learningClassIterator.GetIndex()));
			}

			++learningClassIterator;
		}

		LOG4CXX_INFO(logger, "Learning class loaded from " << training_classes_filenames[i]);
	}
}

void
NeuralNetworkPixelClassifiers
::build_training_sets()
{
	log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("main"));

	unsigned int total_number_of_pixels = 0;
	{
		unsigned int size;
		for(int i = 0; i < m_NumberOfClasses; ++i)
		{
			size = m_TrainingClasses[i]->size();
			total_number_of_pixels += size;

			LOG4CXX_DEBUG(logger, "Class #" << i << ": " << size << " elements");
		}
	}

	// Creating one data set that will be used to initialized the others
	TrainingSet * training_data = fann_create_train(total_number_of_pixels, m_NumberOfComponentsPerPixel, 1);
	fann_type **training_data_input_it = training_data->input;
	fann_type **training_data_output_it = training_data->output;

	for(int i = 0; i < m_NumberOfClasses; ++i)
	{
		boost::shared_ptr< TrainingClass > current_class = m_TrainingClasses[i];

		TrainingClass::const_iterator current_raw_class_it = current_class->begin(), current_raw_class_end = current_class->end();

		while(current_raw_class_it != current_raw_class_end)
		{
			// Copying the features
			std::copy(
					(*current_raw_class_it).GetDataPointer(), 
					(*current_raw_class_it).GetDataPointer() + m_NumberOfComponentsPerPixel, 
					*training_data_input_it
				);

			// Don't care about the outpur right now
			**training_data_output_it = 0;

			++current_raw_class_it;
			++training_data_input_it;
			++training_data_output_it;
		}
	}

	// Storing the first one
	this->m_TrainingSets.push_back( boost::shared_ptr< TrainingSet >( training_data, fann_destroy_train ) );

	// Storing copies of the first one
	for(int i = 1; i < m_NumberOfClassifiers; ++i) {
		this->m_TrainingSets.push_back( boost::shared_ptr< TrainingSet >( fann_duplicate_train_data(training_data), fann_destroy_train ) );
	}

	// Set the desired output of a class to 1 in the dataset representing this class
	int class_start = 0, current_class_size;
	for(int i = 0; i < m_NumberOfClassifiers; ++i) {
		current_class_size = this->m_TrainingClasses[i]->size();

		fann_type *output_start = *(m_TrainingSets[i]->output) + class_start;

		std::fill(output_start, output_start + current_class_size, 1);

		class_start += current_class_size;
	}
}

void
NeuralNetworkPixelClassifiers
::create_and_train_neural_networks( const std::vector< unsigned int > hidden_layers, const float learning_rate, const unsigned int max_epoch, const float mse_target )
{
	log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("main"));

	for(int i = 0; i < m_NumberOfClassifiers; ++i)
	{
		NeuralNetwork* ann = fann_create_standard_array(hidden_layers.size(), hidden_layers.data());
		fann_set_activation_function_hidden(ann, FANN_SIGMOID);
		fann_set_activation_function_output(ann, FANN_SIGMOID);
		fann_set_train_stop_function(ann, FANN_STOPFUNC_MSE);
		fann_set_learning_rate(ann, learning_rate);

		m_NeuralNetworks.push_back( boost::shared_ptr< NeuralNetwork >( ann, fann_destroy ) );
	}

	#pragma omp parallel for
	for(int i = 0; i < m_NumberOfClassifiers; ++i)
	{
		LOG4CXX_INFO(logger, "Training ann #" << i);

		boost::shared_ptr<NeuralNetwork> current_neural_network = m_NeuralNetworks[i];
		boost::shared_ptr<TrainingSet> current_training_set = m_TrainingSets[i];

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

void
NeuralNetworkPixelClassifiers
::save_neural_networks(const std::string dir)
{
	log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("main"));
	LOG4CXX_INFO(logger, "Saving neural networks in " << dir);

	for(int i = 0; i < this->m_NumberOfClassifiers; ++i) {
		std::ostringstream filename;
		filename << std::setfill('0') << std::setw(6) << (i+1) << ".ann";

		boost::filesystem::path path = boost::filesystem::path(dir) / filename.str();

		if(0 != fann_save(this->m_NeuralNetworks[i].get(), path.native().c_str())) {
			throw std::runtime_error("Cannot save neural network in " + path.native());
		}
	}
}

void
NeuralNetworkPixelClassifiers
::load_neural_networks(const std::string dir)
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

	this->m_NumberOfClassifiers = config_files.size();

	for(std::vector<std::string>::const_iterator it = config_files.begin(); it != config_files.end(); ++it) {
		LOG4CXX_INFO(logger, "Loading neural network from " << *it);

		NeuralNetwork* ann = fann_create_from_file(it->c_str());

		if(ann == NULL)
			throw std::runtime_error("Cannot load neural network from " + *it);

		this->m_NeuralNetworks.push_back( boost::shared_ptr< NeuralNetwork >( ann, fann_destroy ) );
	}

	this->m_NumberOfClasses = (this->m_NumberOfClassifiers == 1 ? 2 : this->m_NumberOfClassifiers);
	this->m_NumberOfComponentsPerPixel = fann_get_num_input(this->m_NeuralNetworks.front().get());

	LOG4CXX_INFO(logger, "Number of classes: " << this->m_NumberOfClasses);
	LOG4CXX_INFO(logger, "Number of components per pixel: " << this->m_NumberOfComponentsPerPixel);
}
