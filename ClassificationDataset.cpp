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

ClassificationDataset::ClassificationDataset(typename FeaturesImage::Pointer image, const std::vector< std::string > &class_filenames)
{
	init(class_filenames.size());

	load_image(image, class_filenames);
}

ClassificationDataset::ClassificationDataset(const std::vector< std::string > &images_filenames, const std::vector< std::string > &classes_filenames)
{
	log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("main"));

	const int number_of_classes = classes_filenames.size() / images_filenames.size();

	init(number_of_classes);

	for(int i = 0; i < images_filenames.size(); ++i) {
		LOG4CXX_INFO(logger, "Loading image #" << i << " from " << images_filenames[i]);

		std::vector< std::string > training_classes(
			classes_filenames.begin() + i * number_of_classes,
			classes_filenames.begin() + (i+1) * number_of_classes
		);

		load_image(images_filenames[i], training_classes);
	}
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

		// Load the class image
		try {
			class_image = ImageLoader::load(class_filenames[i]);
		} catch (ImageLoadingException & ex) {
			throw ClassificationDatasetException(ex.what());
		} 

		// check the class image dimensions
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

				DataType values(raw_values.GetDataPointer(), raw_values.GetDataPointer() + m_DataLength);

				current_class.push_back(values);
			}

			++classIterator;
		}

		LOG4CXX_INFO(logger, "Learning class loaded from " << class_filenames[i]);
	}
}

const ClassificationDataset::Class& ClassificationDataset::getClass(const int c)
{
	return m_Classes[c];
}

int ClassificationDataset::getNumberOfClasses() const
{
	return m_NumberOfClasses;
}

int ClassificationDataset::getDataLength() const
{
	return m_DataLength;
}
