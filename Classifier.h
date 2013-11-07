#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <vector>

template <typename TInputValueType>
class Classifier
{
public:
	typedef TInputValueType InputValueType;

	virtual void load(const std::string path) = 0;
	virtual void save(const std::string path) = 0;

	virtual std::vector<float> classify(const std::vector< InputValueType >&) const = 0;
};

#endif /* CLASSIFIER_H */
