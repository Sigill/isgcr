#include "LibSVMClassificationDataset.h"
//#include <fstream> // XXX
#include <iostream>

LibSVMClassificationDataset::LibSVMClassificationDataset(ClassificationDataset<double> &classificationDataset)
{
	m_InputSize = classificationDataset.getInputSize();
	const int numberOfInputs = classificationDataset.getNumberOfInputs();
	const int numberOfInputValues = (m_InputSize + 1) * numberOfInputs;

	prob.l = numberOfInputs;

	try {
		prob.x = new svm_node*[numberOfInputs];
		prob.y = new double[numberOfInputs];
		x_space = boost::shared_array<struct svm_node>(new svm_node[numberOfInputValues]);
	} catch (std::bad_alloc& ba) {
		if(NULL == prob.y)
			delete[] prob.x;

		throw LibSVMClassificationDatasetException("Cannot allocate memory.");
	}

	// XXX
	//std::ofstream file;
	//file.open ("haralick.scale");

	int globalInputId = 0, globalInputValueId = 0;
	for(int i = 0; i < classificationDataset.getNumberOfClasses(); ++i)
	{
		const ClassificationDataset<double>::Class &c = classificationDataset.getClass(i);

		for(int inputId = 0; inputId < c.size(); ++inputId, ++globalInputId)
		{
			prob.y[globalInputId] = i+1; // The class
			prob.x[globalInputId] = &x_space[globalInputValueId];
			//file << i+1; // XXX

			const ClassificationDataset<double>::InputType &input = c[inputId];

			for(int inputValueId = 0; inputValueId < classificationDataset.getInputSize(); ++inputValueId, ++globalInputValueId)
			{
				x_space[globalInputValueId].index = inputValueId + 1;
				x_space[globalInputValueId].value = input[inputValueId];

				//file << " " << (inputValueId+1) << ":" << input[inputValueId]; // XXX
			}
			//file << std::endl; // XXX

			x_space[globalInputValueId].index = -1;
			++globalInputValueId;
		}
	}

	//file.close(); // XXX
}

LibSVMClassificationDataset::~LibSVMClassificationDataset()
{
	delete[] prob.x;
	delete[] prob.y;
}

svm_problem* LibSVMClassificationDataset::getProblem()
{
	return &prob;
}

int LibSVMClassificationDataset::getInputSize() const
{
	return m_InputSize;
}
