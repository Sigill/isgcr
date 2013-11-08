#include "LibSVMClassificationDataset.h"

LibSVMClassificationDataset::LibSVMClassificationDataset(ClassificationDataset<double> &classificationDataset)
{
	const int numberOfInputs = classificationDataset.getNumberOfInputs();
	const int numberOfInputValues = classificationDataset.getInputSize() * numberOfInputs;

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

	int globalInputId = 0, globalInputValueId = 0;
	for(int i = 0; i < classificationDataset.getNumberOfClasses(); ++i)
	{
		const ClassificationDataset<double>::Class &c = classificationDataset.getClass(i);

		for(int inputId = 0; inputId < c.size(); ++inputId, ++globalInputId)
		{
			prob.y[globalInputId] = i+1; // The class
			prob.x[globalInputId] = &x_space[globalInputValueId];

			const ClassificationDataset<double>::InputType &input = c[inputId];

			for(int inputValueId = 0; inputValueId < classificationDataset.getInputSize(); ++inputValueId, ++globalInputValueId)
			{
				x_space[globalInputValueId].index = inputValueId;
				x_space[globalInputValueId].value = input[inputValueId];
			}
		}
	}
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
