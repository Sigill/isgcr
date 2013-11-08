#ifndef FANNCLASSIFICATIONDATASET_H
#define FANNCLASSIFICATIONDATASET_H

#include <boost/shared_ptr.hpp>
#include <vector>
#include <stdexcept>
#include <utility>
#include <doublefann.h>

#include "ClassificationDataset.h"

class FannClassificationDatasetException : public std::runtime_error
{
public:
	FannClassificationDatasetException ( const std::string &err ) : std::runtime_error(err) {}
};

class FannClassificationDataset
{
public:
	typedef struct fann_train_data FannDataset;
	typedef std::vector< boost::shared_ptr< FannDataset > > FannDatasetVector;

private:
	typedef std::vector< boost::shared_ptr< FannDataset > > Container;

public:
	FannClassificationDataset(ClassificationDataset<fann_type> &classificationDataset);
	FannClassificationDataset(Container &sets, const int featuresLength);
	~FannClassificationDataset();

	std::pair< boost::shared_ptr< FannClassificationDataset >, boost::shared_ptr< FannClassificationDataset > > split(const float ratio) const;

	const int getNumberOfDatasets() const;
	const int getInputSize() const;
	void shuffle();
	FannDataset* getSet(const int i) const;

private:
	int m_InputSize;
	Container m_Container;
};

#endif /* FANNCLASSIFICATIONDATASET_H */
