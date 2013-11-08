#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <vector>
#include <string>

template <typename TInputValueType>
class Classifier
{
public:
	typedef TInputValueType InputValueType;

	//Classifier(const unsigned int inputSize, const unsigned int numberOfClasses);

	virtual void load(const std::string path) = 0;
	virtual void save(const std::string path) = 0;

	virtual std::vector<float> classify(const std::vector< InputValueType >&) const = 0;

	unsigned int getInputSize();
	unsigned int getNumberOfClasses();

protected:
	unsigned int m_InputSize, m_NumberOfClasses;
};

#ifndef MANUAL_INSTANTIATION
#include "Classifier.cpp"
#endif

#endif /* CLASSIFIER_H */
