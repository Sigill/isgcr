#ifndef LIBSVMCLASSIFICATIONDATASET_H
#define LIBSVMCLASSIFICATIONDATASET_H

#include "ClassificationDataset.h"

#include <stdexcept>
#include <log4cxx/logger.h>
#include <libsvm/svm.h>
#include <boost/shared_array.hpp>

class LibSVMClassificationDatasetException : public std::runtime_error
{
public:
	LibSVMClassificationDatasetException ( const std::string &err ) : std::runtime_error(err) {}
};

class LibSVMClassificationDataset
{
public:
	LibSVMClassificationDataset(ClassificationDataset<double> &classificationDataset);
	~LibSVMClassificationDataset();

	svm_problem* getProblem();

	int getInputSize() const;

private:
	boost::shared_array<struct svm_node> x_space;
	struct svm_problem prob;
	int m_InputSize;
};

#endif /* LIBSVMCLASSIFICATIONDATASET_H */
