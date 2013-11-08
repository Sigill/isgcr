#ifndef SVMPIXELCLASSIFIER_H
#define SVMPIXELCLASSIFIER_H

#include "Classifier.h"

class SVMPixelClassifier : public Classifier<double>
{
public:
	void load(const std::string path);
	void save(const std::string path);

	std::vector<float> classify(const std::vector< InputValueType >&) const;

};

#endif /* SVMPIXELCLASSIFIER_H */
