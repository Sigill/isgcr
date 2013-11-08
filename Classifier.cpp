#include "Classifier.h"

/*
template <typename TInputValueType>
unsigned int Classifier<TInputValueType>::Classifier(const unsigned int inputSize, const unsigned int numberOfClasses) :
	m_InputSize(inputSize),
	m_NumberOfClasses(numberOfClasses)
{}
*/

template <typename TInputValueType>
unsigned int Classifier<TInputValueType>::getInputSize()
{
	return m_InputSize;
}

template <typename TInputValueType>
unsigned int Classifier<TInputValueType>::getNumberOfClasses()
{
	return m_NumberOfClasses;
}
