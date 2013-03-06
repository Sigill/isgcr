#include "ClassificationDataset.h"
#include "image_loader.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "log4cxx/logger.h"

void ClassificationDataset::init(const int number_of_classes)
{
	m_NumberOfClasses = number_of_classes;
	m_Classes = ClassVector(number_of_classes, Class());
	m_DataLength = 0;
}

void ClassificationDataset::load_image(const std::string image_filename, const std::vector< std::string > class_filenames)
{
	typename itk::ImageFileReader< FeaturesImage >::Pointer reader = itk::ImageFileReader< FeaturesImage >::New();
	reader->SetFileName(image_filename);

	try {
		reader->Update();
	} catch( itk::ExceptionObject &ex ) {
		std::stringstream err;
		err << "ITK is unable to load the image \"" << image_filename << "\" (" << ex.what() << ")";

		throw ClassificationDatasetException(err.str());
	}

	this->load_image(reader->GetOutput(), class_filenames);
}

void ClassificationDataset::load_image(typename FeaturesImage::Pointer image, const std::vector< std::string > class_filenames)
{
	log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("main"));

	if(m_DataLength == 0)
		m_DataLength = image->GetNumberOfComponentsPerPixel();
	else if(m_DataLength != image->GetNumberOfComponentsPerPixel())
		throw ClassificationDatasetException("The image has a number of components which is unexpected.");

	/**
	  * Loading the classes.
	  * For each class we build a vector of the pixels
	  * to be used during learning.
	  */
	for(int i = 0; i < m_NumberOfClasses; ++i)
	{
		LOG4CXX_INFO(logger, "Loading class from " << class_filenames[i]);

		ImageType::Pointer class_image;
		try {
			class_image = ImageLoader::load(class_filenames[i]);
		} catch (ImageLoadingException & ex) {
			throw ClassificationDatasetException(ex.what());
		} 

		if(class_image->GetLargestPossibleRegion().GetSize() != image->GetLargestPossibleRegion().GetSize()) {
			std::stringstream err;
			err << "The dimensions of the class image \"" << class_filenames[i] << "\" (" << class_image->GetLargestPossibleRegion().GetSize()
			    << ") differs from the dimensions of the image (" << image->GetLargestPossibleRegion().GetSize() << ")";

			throw ClassificationDatasetException(err.str());
		}

		Class &current_class = m_Classes[i];

		typename itk::ImageRegionConstIteratorWithIndex< ImageType > classIterator(class_image, class_image->GetLargestPossibleRegion());
		while(!classIterator.IsAtEnd())
		{
			if(255 == classIterator.Get()) {
				const typename FeaturesImage::PixelType raw_values = image->GetPixel(classIterator.GetIndex());

				DataType values;
				for(int i = 0; i < m_DataLength; ++i) {
					values.push_back(raw_values[i]);
				}

				current_class.push_back(values);
			}

			++classIterator;
		}

		LOG4CXX_INFO(logger, "Learning class loaded from " << class_filenames[i]);
	}
}

boost::shared_ptr< typename ClassificationDataset::FannDatasetVector >
ClassificationDataset::build_fann_binary_sets()
{
	log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("main"));

	unsigned int total_number_of_elements = 0;
	{
		for(int i = 0; i < m_NumberOfClasses; ++i)
		{
			total_number_of_elements += m_Classes[i].size();

			LOG4CXX_DEBUG(logger, "Class #" << i << ": " << m_Classes[i].size() << " elements");
		}
	}

	// Creating one data set that will be used to initialized the others
	FannDataset *training_data = fann_create_train(total_number_of_elements, m_DataLength, 1);
	fann_type **training_data_input_it = training_data->input;
	fann_type **training_data_output_it = training_data->output;

	for(int i = 0; i < m_NumberOfClasses; ++i)
	{
		Class &current_class = m_Classes[i];

		Class::const_iterator current_raw_class_it = current_class.begin(), current_raw_class_end = current_class.end();

		while(current_raw_class_it != current_raw_class_end)
		{
			// Copying the features
			std::copy(
					current_raw_class_it->begin(),
					current_raw_class_it->end(), 
					*training_data_input_it
					);

			// Don't care about the output right now
			**training_data_output_it = 0;

			++current_raw_class_it;
			++training_data_input_it;
			++training_data_output_it;
		}
	}

	boost::shared_ptr< FannDatasetVector > fannDatasets(new FannDatasetVector);

	// Storing the first one
	fannDatasets->push_back( boost::shared_ptr< FannDataset >( training_data, fann_destroy_train ) );

	const int numberOfClassifiers = (m_NumberOfClasses == 2 ? 1 : m_NumberOfClasses);

	// Storing copies of the first one
	for(int i = 1; i < numberOfClassifiers; ++i) {
		fannDatasets->push_back( boost::shared_ptr< FannDataset >( fann_duplicate_train_data(training_data), fann_destroy_train ) );
	}

	// Set the desired output of a class to 1 in the dataset representing this class
	int class_start = 0, current_class_size;
	for(int i = 0; i < numberOfClassifiers; ++i) {
		current_class_size = m_Classes[i].size();

		fann_type *output_start = *(fannDatasets->operator[](i)->output) + class_start;

		std::fill(output_start, output_start + current_class_size, 1);

		class_start += current_class_size;
	}

	return fannDatasets;
}
