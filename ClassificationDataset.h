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

	typedef struct fann_train_data FannDataset;
	typedef std::vector< boost::shared_ptr< FannDataset > > FannDatasetVector;

	void init(const int number_of_classes);

	void load_image(const std::string image_filename, const std::vector< std::string > class_filenames);
	void load_image(typename FeaturesImage::Pointer image, const std::vector< std::string > class_filenames);

	boost::shared_ptr< FannDatasetVector > build_fann_binary_training_sets();

	int getNumberOfClasses() { return m_NumberOfClasses; }
	int getDataLength() { return m_DataLength; }

private:
	typedef std::vector< DataType > Class;
	typedef std::vector< Class > ClassVector;

	int m_DataLength;
	int m_NumberOfClasses;
	ClassVector m_Classes;
};

#endif /* CLASSIFICATIONDATASET_H */
