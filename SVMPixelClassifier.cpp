#include "SVMPixelClassifier.h"
#include <libsvm/svm.h>
#include "log4cxx/logger.h"
#include <boost/filesystem.hpp>
#include <iostream>

void SVMPixelClassifier::load(const std::string dir)
{
	boost::filesystem::path path = boost::filesystem::path(dir) / "svm.model";
	svm_model *m;
	if((m = svm_load_model(path.native().c_str())) == 0)
		throw std::runtime_error("Cannot load the SVM from " + path.native());
	else
		model = boost::shared_ptr<struct svm_model>(m, svm_free_model_content);

	m_NumberOfClasses = svm_get_nr_class(model.get());
	m_InputSize = 0;

	if(svm_check_probability_model(model.get()) == 0)
		throw std::runtime_error("Model does not support probabiliy estimates.");
}

void SVMPixelClassifier::save(const std::string dir)
{
	boost::filesystem::path path = boost::filesystem::path(dir) / "svm.model";

	if(0 != svm_save_model(path.native().c_str(), model.get()))
	{
		throw std::runtime_error("Cannot save the SVM in " + path.native());
	}
}

std::vector<float> SVMPixelClassifier::classify(const std::vector< InputValueType > &input) const
{
	double estimates[m_NumberOfClasses];
	struct svm_node x[input.size()+1];
	for(int i = 0; i < input.size(); ++i)
	{
		x[i].index = i+1;
		x[i].value = input[i];
	}
	x[input.size()].index = -1;

	svm_predict_probability(model.get(), x, estimates);

	std::vector<float> output(m_NumberOfClasses, 0);
	for(int i = 0; i < m_NumberOfClasses; ++i)
	{
		output[i] = (float)estimates[i];
	}

	return output;
}

bool SVMPixelClassifier::train(LibSVMClassificationDataset *trainingSet)
{
	log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("main"));

	struct svm_parameter param;
	param.svm_type = C_SVC;
	param.kernel_type = RBF;
	param.degree = 3;
	param.gamma = 1.0 / (trainingSet->getInputSize() + 1);        // 1/num_features
	param.coef0 = 0;
	param.nu = 0.5;
	param.cache_size = 100;
	param.C = 1;
	param.eps = 1e-3;
	param.p = 0.1;
	param.shrinking = 1;
	param.probability = 1; // Compute probabilities
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;

	const char *error_msg;
	error_msg = svm_check_parameter(trainingSet->getProblem(), &param);

	if(error_msg) {
		LOG4CXX_FATAL(logger, error_msg);
		return false;
	}
	svm_set_print_string_function(NULL);

	struct svm_model *m = svm_train(trainingSet->getProblem(), &param);
	boost::filesystem::path ph = boost::filesystem::temp_directory_path() / boost::filesystem::unique_path();
	if(0 != svm_save_model(ph.native().c_str(), m))
		throw std::runtime_error("Cannot temporary save the SVM in " + ph.native());

	svm_free_and_destroy_model(&m);

	if((m = svm_load_model(ph.native().c_str())) == 0)
		throw std::runtime_error("Cannot load the temporary SVM from " + ph.native());
	else
		model = boost::shared_ptr<struct svm_model>(m, svm_free_model_content);

	boost::filesystem::remove(ph);

	//model = boost::shared_ptr<struct svm_model>(svm_train(trainingSet->getProblem(), &param), svm_free_model_content);

	m_NumberOfClasses = svm_get_nr_class(model.get());

	return true;
}

