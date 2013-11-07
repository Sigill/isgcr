#ifndef CLASSIFICATIONDATASET_H
#define CLASSIFICATIONDATASET_H

#include "common.h"

#include <boost/shared_ptr.hpp>
#include <vector>
#include <string>
#include <stdexcept>
#include <utility>

class ClassificationDatasetException : public std::runtime_error
{
public:
	ClassificationDatasetException ( const std::string &err ) : std::runtime_error(err) {}
};


/**
 * \class ClassificationDataset
 *
 * \brief Contains several lists of patterns, each list representing a class.
 *
 * The template parameter correspond to the data type of the patterns (float, double).
 */
template <typename TInputValueType>
class ClassificationDataset
{
public:
	typedef TInputValueType InputValueType;

	/** InputType represents a texture feature. */
	typedef std::vector< InputValueType > InputType;

	/** A Class is a set of texture features. */
	typedef std::vector< InputType > Class;

private:
	/** Datatype representing a list of class. */
	typedef std::vector< Class > ClassVector;

public:
	/**
	 * Build a ClassificationDataset from a single (already loaded) image.
	 * The number of classes will be equal to the number of masks.
	 *
	 * @param image The image that holds the features.
	 * @param class_filenames The filenames of the masks.
	 */
	ClassificationDataset(typename FeaturesImage::Pointer image, const std::vector< std::string > &class_filenames);

	/**
	 * Build a ClassificationDataset from a a list of images (and theirs associated masks).
	 * The images must have the same number of components per pixel. The list of masks is 
	 * packed by image. The algorithm expect them to be properly ordered.
	 *
	 * @param image The filenames of the images that holds the features.
	 * @param class_filenames The filenames of the masks.
	 */
	ClassificationDataset(const std::vector< std::string > &image_filenames, const std::vector< std::string > &class_filenames);

	ClassificationDataset(ClassVector *classes, const int number_of_classes, const int input_size);

	/**
	 * Returns a reference on a class.
	 *
	 * @param c The index of the class.
	 */
	const Class& getClass(const int c) const;

	/** The number of classes. */
	int getNumberOfClasses() const;

	/** The length of a pattern. */
	int getInputSize() const;

	std::pair< boost::shared_ptr< ClassificationDataset<InputValueType> >, boost::shared_ptr< ClassificationDataset<InputValueType> > > split(const float ratio) const;

	void shuffle();

private:
	void init(const int number_of_classes);
	void load_image(const std::string image_filename, const std::vector< std::string > class_filenames);
	void load_image(typename FeaturesImage::Pointer image, const std::vector< std::string > class_filenames);

	int m_InputSize;
	int m_NumberOfClasses;
	ClassVector m_Classes;
};

#ifndef MANUAL_INSTANTIATION
#include "ClassificationDataset.cpp"
#endif

#endif /* CLASSIFICATIONDATASET_H */
