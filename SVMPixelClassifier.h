#ifndef SVMPIXELCLASSIFIER_H
#define SVMPIXELCLASSIFIER_H

#include "Classifier.h"
#include "LibSVMClassificationDataset.h"
#include <boost/shared_array.hpp>

class SVMPixelClassifier : public Classifier<double>
{
public:
	void load(const std::string dir);
	void save(const std::string dir);

	std::vector<float> classify(const std::vector< InputValueType >&) const;

	bool train(LibSVMClassificationDataset *trainingSet);

private:
	boost::shared_ptr<struct svm_model> model;
};

#endif /* SVMPIXELCLASSIFIER_H */
