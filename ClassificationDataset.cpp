#include "ClassificationDataset.h"
#include "image_loader.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "log4cxx/logger.h"
#include <cstdlib> // rand()

template <typename TInputValueType>
void ClassificationDataset<TInputValueType>::init(const int number_of_classes)
{
	m_NumberOfClasses = number_of_classes;
	m_Classes = ClassVector(number_of_classes, Class());
	m_InputSize = 0;
}

template <typename TInputValueType>
ClassificationDataset<TInputValueType>::ClassificationDataset(typename FeaturesImage::Pointer image, const std::vector< std::string > &class_filenames)
{
	init(class_filenames.size());

	load_image(image, class_filenames);
}

template <typename TInputValueType>
ClassificationDataset<TInputValueType>::ClassificationDataset(const std::vector< std::string > &images_filenames, const std::vector< std::string > &classes_filenames)
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

template <typename TInputValueType>
ClassificationDataset<TInputValueType>::ClassificationDataset(ClassVector *classes, const int number_of_classes, const int input_size) :
	m_NumberOfClasses(number_of_classes),
	m_InputSize(input_size),
	m_Classes(*classes)
{}

template <typename TInputValueType>
void ClassificationDataset<TInputValueType>::load_image(const std::string image_filename, const std::vector< std::string > class_filenames)
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

template <typename TInputValueType>
void ClassificationDataset<TInputValueType>::load_image(typename FeaturesImage::Pointer image, const std::vector< std::string > class_filenames)
{
	log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("main"));

	if(m_InputSize== 0)
		m_InputSize= image->GetNumberOfComponentsPerPixel();
	else if(m_InputSize != image->GetNumberOfComponentsPerPixel())
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

				InputType values(raw_values.GetDataPointer(), raw_values.GetDataPointer() + m_InputSize);

				current_class.push_back(values);
			}

			++classIterator;
		}

		LOG4CXX_INFO(logger, "Learning class loaded from " << class_filenames[i]);
	}
}

template <typename TInputValueType>
const typename ClassificationDataset<TInputValueType>::Class& ClassificationDataset<TInputValueType>::getClass(const int c) const
{
	return m_Classes[c];
}

template <typename TInputValueType>
int ClassificationDataset<TInputValueType>::getNumberOfClasses() const
{
	return m_NumberOfClasses;
}

template <typename TInputValueType>
int ClassificationDataset<TInputValueType>::getNumberOfInputs() const
{
	int total_number_of_elements = 0;
	for(int i = 0; i < m_NumberOfClasses; ++i)
		total_number_of_elements += getClass(i).size();

	return total_number_of_elements;
}

template <typename TInputValueType>
void ClassificationDataset<TInputValueType>::checkValid() const
{
	log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("main"));

	if(0 == m_NumberOfClasses)
		throw ClassificationDatasetException("This dataset does not contain any class.");

	for(int i = 0; i < m_NumberOfClasses; ++i)
	{
		if(getClass(i).empty()) {
			std::stringstream err;
			err << "The class #" << i << " is empty.";
			throw ClassificationDatasetException(err.str());
		}
	}
}

template <typename TInputValueType>
int ClassificationDataset<TInputValueType>::getInputSize() const
{
	return m_InputSize;
}

template <typename TInputValueType>
std::pair< boost::shared_ptr< ClassificationDataset<TInputValueType> >, boost::shared_ptr< ClassificationDataset<TInputValueType> > >
ClassificationDataset<TInputValueType>::split(const float ratio) const
{
	ClassVector cv1(m_NumberOfClasses, Class()),
	            cv2(m_NumberOfClasses, Class());

	for(int i = 0; i < m_NumberOfClasses; ++i)
	{
		const int number_of_elements = getClass(i).size();
		const int first_set_size     = round( number_of_elements * ratio );
		const int second_set_size    = number_of_elements - first_set_size;

		if((0 == first_set_size) || (0 == second_set_size))
			throw ClassificationDatasetException("Cannot split this ClassificationDataset. The ratio will ends-up generating an empty set.");

		cv1[i].insert(cv1[i].end(), getClass(i).begin(), getClass(i).begin() + first_set_size);
		cv2[i].insert(cv2[i].end(), getClass(i).begin() + first_set_size, getClass(i).end());
	}

	return std::make_pair(
		new ClassificationDataset(&cv1, m_NumberOfClasses, m_InputSize),
		new ClassificationDataset(&cv2, m_NumberOfClasses, m_InputSize)
		);
}

template <typename TInputValueType>
void ClassificationDataset<TInputValueType>::shuffle()
{
	for(int i = 0; i < m_NumberOfClasses; ++i)
	{
		Class &c = m_Classes[i];
		for(int j = c.size() - 1; j > 0; --j)
		{
			c[j].swap(c[std::rand() % (j+1)]);
		}
	}
}

