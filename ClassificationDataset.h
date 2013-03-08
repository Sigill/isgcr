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

class ClassificationDataset
{
public:
	typedef std::vector< fann_type > DataType;
	typedef std::vector< DataType > Class;

	ClassificationDataset(typename FeaturesImage::Pointer image, const std::vector< std::string > &class_filenames);
	ClassificationDataset(const std::vector< std::string > &image_filenames, const std::vector< std::string > &class_filenames);

	const Class& getClass(const int c);
	int getNumberOfClasses() const;
	int getDataLength() const;

private:
	typedef std::vector< Class > ClassVector;

	void init(const int number_of_classes);
	void load_image(const std::string image_filename, const std::vector< std::string > class_filenames);
	void load_image(typename FeaturesImage::Pointer image, const std::vector< std::string > class_filenames);

	int m_DataLength;
	int m_NumberOfClasses;
	ClassVector m_Classes;
};

#endif /* CLASSIFICATIONDATASET_H */
