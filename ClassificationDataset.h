#ifndef CLASSIFICATIONDATASET_H
#define CLASSIFICATIONDATASET_H

#include "common.h"

#include "doublefann.h"
#include <boost/shared_ptr.hpp>
#include <vector>
#include <string>
#include <stdexcept>

class ClassificationDatasetException : public std::runtime_error
{
public:
	ClassificationDatasetException ( const std::string &err ) : std::runtime_error(err) {}
};

/**
 * \class ClassificationDataset
 *
 * \brief Contains several lists of patterns, each list representing a class.
 * It is designed to be built on an image (or several images) based on a set
 * of masks (each mask defining a class).
 */
class ClassificationDataset
{
public:
	/** Datatype represents a texture feature. */
	typedef std::vector< fann_type > DataType;

	/** A Class is a set of texture features. */
	typedef std::vector< DataType > Class;

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

	/**
	 * Returns a reference on a class.
	 *
	 * @param c The index of the class.
	 */
	const Class& getClass(const int c);

	/** The number of classes. */
	int getNumberOfClasses() const;

	/** The length of a pattern. */
	int getDataLength() const;

private:
	/** Datatype representing a list of class. */
	typedef std::vector< Class > ClassVector;

	void init(const int number_of_classes);
	void load_image(const std::string image_filename, const std::vector< std::string > class_filenames);
	void load_image(typename FeaturesImage::Pointer image, const std::vector< std::string > class_filenames);

	int m_DataLength;
	int m_NumberOfClasses;
	ClassVector m_Classes;
};

#endif /* CLASSIFICATIONDATASET_H */
