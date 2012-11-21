#include "NeuralNetworkPixelClassifiers.h"
#include "image_loader.h"

#include "itkImageRegionConstIteratorWithIndex.h"

#include "log4cxx/logger.h"

void
NeuralNetworkPixelClassifiers
::load_training_sets(const std::vector< std::string > filenames, typename FeaturesImage::Pointer featuresImage)
{
	log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("main"));

	m_TrainingSets.clear();

	m_NumberOfClasses = filenames.size();

	boost::shared_ptr<TrainingClassVector> raw_learning_classes(new TrainingClassVector());

	/**
	  * Loading the classes.
	  * For each learning class we build a vector of the pixels
	  * to be used during learning.
	  */
	for(int i = 0; i < m_NumberOfClasses; ++i)
	{
		ImageType::Pointer image;
		try {
			image = ImageLoader::load(filenames[i]);
		} catch (ImageLoadingException & ex) {
			throw LearningClassException(ex.what());
		} 

		if(image->GetLargestPossibleRegion().GetSize() != featuresImage->GetLargestPossibleRegion().GetSize())
		{
			std::stringstream err;
			err << "The dimensions of " << filenames[i] << "(" << image->GetLargestPossibleRegion().GetSize() << ") differs from the dimensions of the image (" << featuresImage->GetLargestPossibleRegion().GetSize() << ")";
			throw LearningClassException(err.str());
		}

		boost::shared_ptr< TrainingClass > current_class(new TrainingClass);


		typename itk::ImageRegionConstIteratorWithIndex< ImageType > learningClassIterator(image, image->GetLargestPossibleRegion());
		while(!learningClassIterator.IsAtEnd())
		{
			if(255 == learningClassIterator.Get())
			{
				current_class->push_back(featuresImage->GetPixel(learningClassIterator.GetIndex()));
			}

			++learningClassIterator;
		}

		LOG4CXX_INFO(logger, "Learning class loaded from " << filenames[i]);

		raw_learning_classes->push_back(current_class);
	}


	unsigned int total_number_of_pixels = 0;
	{
		unsigned int size;
		for(int i = 0; i < m_NumberOfClasses; ++i)
		{
			size = raw_learning_classes->operator[](i)->size();
			total_number_of_pixels += size;

			LOG4CXX_DEBUG(logger, "Class #" << i << ": " << size << " elements");
		}
	}

	m_NumberOfComponentsPerPixel = featuresImage->GetNumberOfComponentsPerPixel();

	// Creating one data set that will be used to initialized the others
	TrainingSet * training_data = fann_create_train(total_number_of_pixels, m_NumberOfComponentsPerPixel, 1);
	fann_type **training_data_input_it = training_data->input;
	fann_type **training_data_output_it = training_data->output;

	for(int i = 0; i < m_NumberOfClasses; ++i)
	{
		boost::shared_ptr< TrainingClass > current_class = raw_learning_classes->operator[](i);

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
	m_TrainingSets.push_back( boost::shared_ptr< TrainingSet >( training_data, fann_destroy_train ) );

	// Storing copies of the first one
	for(int i = 1; i < m_NumberOfClasses; ++i)
	{
		m_TrainingSets.push_back( boost::shared_ptr< TrainingSet >( fann_duplicate_train_data(training_data), fann_destroy_train ) );
	}

	// Set the desired output of a class to 1 in the dataset representing this class
	int class_start = 0, current_class_size;
	for(int i = 0; i < m_NumberOfClasses; ++i)
	{
		current_class_size = raw_learning_classes->operator[](i)->size();

		fann_type *output_start = *(m_TrainingSets[i]->output) + class_start;

		std::fill(output_start, output_start + current_class_size, 1);

		class_start += current_class_size;
	}
}

void
NeuralNetworkPixelClassifiers
::create_and_train_neural_networks( const std::vector< int > hidden_layers, const float learning_rate, const unsigned int max_epoch, const float mse_target )
{
	log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("main"));

	for(int i = 0; i < m_NumberOfClasses; ++i)
	{
		NeuralNetwork* ann = fann_create_standard_array(hidden_layers.size(), (const unsigned int*)(hidden_layers.data()));
		fann_set_activation_function_hidden(ann, FANN_SIGMOID);
		fann_set_activation_function_output(ann, FANN_SIGMOID);
		fann_set_train_stop_function(ann, FANN_STOPFUNC_MSE);
		fann_set_learning_rate(ann, learning_rate);

		m_NeuralNetworks.push_back( boost::shared_ptr< NeuralNetwork >( ann, fann_destroy ) );
	}

	#pragma omp parallel for
	for(int i = 0; i < m_NumberOfClasses; ++i)
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
