#include "classification.h"

#include "image_loader.h"

#include "log4cxx/logger.h"

#include <ostream>

typedef itk::ImageRegionConstIteratorWithIndex< ImageType > ConstIterator;

boost::shared_ptr<TrainingClassVector> load_classes(const std::vector< std::string > filenames, NormalizedHaralickImage::Pointer haralickImage) throw(LearningClassException)
{
	log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("main"));

	const unsigned int number_of_classes = filenames.size();

	boost::shared_ptr<TrainingClassVector> raw_learning_classes(new TrainingClassVector());

	for(int i = 0; i < number_of_classes; ++i)
	{
		ImageType::Pointer image;
		try {
			image = ImageLoader::load(filenames[i]);
		} catch (ImageLoadingException & ex) {
			throw LearningClassException(ex.what());
		} 

		if(image->GetLargestPossibleRegion().GetSize() != haralickImage->GetLargestPossibleRegion().GetSize())
		{
			std::stringstream err;
			err << "The dimensions of " << filenames[i] << "(" << image->GetLargestPossibleRegion().GetSize() << ") differs from the dimensions of the image (" << haralickImage->GetLargestPossibleRegion().GetSize() << ")";
			throw LearningClassException(err.str());
		}

		boost::shared_ptr< TrainingClass > current_class(new TrainingClass);

		ConstIterator learningClassIterator(image, image->GetLargestPossibleRegion());
		while(!learningClassIterator.IsAtEnd())
		{
			if(255 == learningClassIterator.Get())
			{
				current_class->push_back(haralickImage->GetPixel(learningClassIterator.GetIndex()));
			}

			++learningClassIterator;
		}

		LOG4CXX_INFO(logger, "Learning class loaded from " << filenames[i]);

		raw_learning_classes->push_back(current_class);
	}

	return raw_learning_classes;
}

boost::shared_ptr<TrainingSetVector> generate_training_sets( boost::shared_ptr<TrainingClassVector> raw_learning_classes)
{
	log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("main"));

	const unsigned int number_of_classes = raw_learning_classes->size();

	unsigned int total = 0;
	{
		unsigned int previous_total;
		for(int i = 0; i < number_of_classes; ++i)
		{
			previous_total = total;
			total += raw_learning_classes->operator[](i)->size();

			LOG4CXX_DEBUG(logger, "Class #" << i << ": " << (total - previous_total) << " elements");
		}
	}

	// Creating one data set that will be used to initialized the others
	TrainingSet * training_data = fann_create_train(total, 8, 1);
	fann_type **training_data_input_it = training_data->input;
	fann_type **training_data_output_it = training_data->output;

	for(int i = 0; i < number_of_classes; ++i)
	{
		boost::shared_ptr< TrainingClass > current_class = raw_learning_classes->operator[](i);

		TrainingClass::const_iterator current_raw_class_it = current_class->begin(), current_raw_class_end = current_class->end();

		while(current_raw_class_it != current_raw_class_end)
		{
			std::copy((*current_raw_class_it).GetDataPointer(), (*current_raw_class_it).GetDataPointer() + 8, *training_data_input_it);

			**training_data_output_it = 0;

			++current_raw_class_it;
			++training_data_input_it;
			++training_data_output_it;
		}
	}

	boost::shared_ptr< TrainingSetVector > learning_classes(new TrainingSetVector());
	// Storing the first one
	learning_classes->push_back( boost::shared_ptr< TrainingSet >( training_data, fann_destroy_train ) );

	// Storing copies of the first one
	for(int i = 1; i < number_of_classes; ++i)
	{
		learning_classes->push_back( boost::shared_ptr< TrainingSet >( fann_duplicate_train_data(training_data), fann_destroy_train ) );
	}

	// Set the desired output of a class to 1 in the dataset representing this class
	int class_start = 0, current_class_size;
	for(int i = 0; i < number_of_classes; ++i)
	{
		current_class_size = raw_learning_classes->operator[](i)->size();

		fann_type *output_start = *(learning_classes->operator[](i)->output) + class_start;

		std::fill(output_start, output_start + current_class_size, 1);

		class_start += current_class_size;
	}

	return learning_classes;
}

boost::shared_ptr< NeuralNetworkVector > train_neural_networks(boost::shared_ptr<TrainingSetVector> training_sets)
{
	log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("main"));

	const unsigned int number_of_classes = training_sets->size();

	boost::shared_ptr< NeuralNetworkVector > networks(new NeuralNetworkVector());

	for(int i = 0; i < number_of_classes; ++i)
	{
		NeuralNetwork* ann = fann_create_standard(3, 8, 3, 1);
		fann_set_activation_function_hidden(ann, FANN_SIGMOID);
		fann_set_activation_function_output(ann, FANN_SIGMOID);
		fann_set_train_stop_function(ann, FANN_STOPFUNC_MSE);
		fann_set_learning_rate(ann, 0.1);

		networks->push_back( boost::shared_ptr< NeuralNetwork >( ann, fann_destroy ) );
	}

#pragma omp parallel for
	for(int i = 0; i < number_of_classes; ++i)
	{
		LOG4CXX_INFO(logger, "Training ann #" << i);

		boost::shared_ptr<NeuralNetwork> current_neural_network = networks->operator[](i);
		boost::shared_ptr<TrainingSet> current_training_set = training_sets->operator[](i);

		fann_shuffle_train_data(current_training_set.get());

		fann_train_on_data(current_neural_network.get(), current_training_set.get(), 1000, 0, 0.0001);

		LOG4CXX_INFO(logger, "MSE for ann #" << i << ": " << fann_get_MSE(current_neural_network.get()));
	}

	return networks;
}

