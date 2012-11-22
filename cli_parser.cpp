#include "cli_parser.h"

#include <boost/filesystem.hpp>
#include <iostream>
#include <sstream>
#include <vector>

#include "ParseUtils.h"


#include "log4cxx/logger.h"

namespace po = boost::program_options;

void validate(boost::any& v, const std::vector<std::string>& values, positive_integer*, int)
{
	po::validators::check_first_occurrence(v);
	const std::string& s = po::validators::get_single_string(values);

	unsigned int value;
	if( ParseUtils::ParseUInt(value, s.data(), 10) && value > 0 )
	{
		v = boost::any(positive_integer(value));
	} else {
		throw po::invalid_option_value(s);
	}
}

void validate(boost::any& v, const std::vector<std::string>& values, strictly_positive_integer*, int)
{
	po::validators::check_first_occurrence(v);
	const std::string& s = po::validators::get_single_string(values);

	unsigned int value;
	if( ParseUtils::ParseUInt(value, s.data(), 10) && value > 0 )
	{
		v = boost::any(strictly_positive_integer(value));
	} else {
		throw po::invalid_option_value(s);
	}
}

std::ostream& operator<< (std::ostream& s, const strictly_positive_integer& v)
{
	return s << v.value;
}

std::ostream& operator<< (std::ostream& s, const positive_integer& v)
{
	return s << v.value;
}

std::ostream &operator<<(std::ostream &out, std::vector< int >& t)
{
	std::stringstream str;
	std::copy(t.begin(), t.end(), std::ostream_iterator<int>(out, ", ") );

	return out;
}

CliParser::CliParser()
{}

int CliParser::parse_argv(int argc, char ** argv)
{
	log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("main"));

	LOG4CXX_INFO(logger, "Parsing command line options");

	po::options_description desc("Command line parameters");
	desc.add_options()
		("help,h",
			"Produce help message")
		("input-image,i",
			po::value< std::string >(&(this->input_image))->required(),
			"Input image (required)")
		("class-image,c",
			po::value< std::vector< std::string > >(&(this->class_images))->required(),
			"Defines a class to be learned from a binary image (at least 2 values required)")
		("export-dir,E",
			po::value< std::string >(&(this->export_dir))->required(),
			"Export directory")
		("export-interval,e",
			po::value< int >(&(this->export_interval))->default_value(0),
			"Export interval during regularization")
		("num-iter,n",
			po::value< int >(&(this->num_iter))->default_value(0),
			"Number of iterations for the regularization")
		("lambda1",
			po::value< double >(&(this->lambda1))->default_value(1.0),
			"Lambda 1 parameter for regularization")
		("lambda2",
			po::value< double >(&(this->lambda2))->default_value(1.0),
			"Lambda 2 parameter for regularization")
		("ann-hidden-layer",
			po::value< std::vector< int > >(&(this->ann_hidden_layers))->multitoken(),
			"Number of neurons per hidden layer (default: one layer of 3 neurons)")
		("ann-learning-rate",
			po::value< float >(&(this->ann_learning_rate))->default_value(0.1),
			"Learning rate of the neural networks")
		("ann-max-epoch",
			po::value< strictly_positive_integer >(&(this->ann_max_epoch))->default_value(strictly_positive_integer(1000)),
			"Maximum number of learning iterations for the neural networks")
		("ann-mse-target",
			po::value< float >(&(this->ann_learning_rate))->default_value(0.0001),
			"Mean squared error targeted by the neural networks learning algorithm")
		;

	po::variables_map vm;

	try {
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);

		// Handling --help before notify() in order to allow ->required()
		// http://stackoverflow.com/questions/5395503/required-and-optional-arguments-using-boost-library-program-options#answer-5517755
		if (vm.count("help")) {
			std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
			std::cout << desc;
			return 0;
		}

		po::notify(vm);
	} catch(po::error &err) {
		LOG4CXX_FATAL(logger, err.what());
		return -1;
	}

	LOG4CXX_INFO(logger, "Input image: " << this->input_image);

	if(this->class_images.size() < 2)
	{
		LOG4CXX_FATAL(logger, "You need to provide at least two learning classes");
		return -1;
	} else {
		LOG4CXX_INFO(logger, this->class_images.size() << " learning classes provided");

		for(int i = 0; i < this->class_images.size(); ++i)
		{
			boost::filesystem::path path(this->input_image);
			if(!boost::filesystem::exists(path))
			{
				LOG4CXX_FATAL(logger, "Cannot load class image: \"" << this->class_images[i] << "\" does not exists");
				return -1;
			}

			LOG4CXX_INFO(logger, "Class " << (i + 1) << ": " << this->class_images[i]);
		}
	}

	if(this->ann_hidden_layers.empty()) {
		this->ann_hidden_layers.push_back(3);
	}

	LOG4CXX_INFO(logger, "Main parameters:");
	LOG4CXX_INFO(logger, "\tExport directory: " << this->export_dir);
	LOG4CXX_INFO(logger, "Regularization parameters:");
	LOG4CXX_INFO(logger, "\tExport interval: " << this->export_interval);
	LOG4CXX_INFO(logger, "\tNumber of iterations: " << this->num_iter);
	LOG4CXX_INFO(logger, "\tLambda2: " << this->lambda2);
	LOG4CXX_INFO(logger, "\tLambda1: " << this->lambda1);
	LOG4CXX_INFO(logger, "Neural networks parameters:");
	{
		std::stringstream m;
		m << this->ann_hidden_layers;
		LOG4CXX_INFO(logger, "\tNumber of hidden neurons per layer: " << m.str());
	}
	LOG4CXX_INFO(logger, "\tLearning rate: " << this->ann_learning_rate);
	LOG4CXX_INFO(logger, "\tMaximum number of iterations: " << this->ann_max_epoch.value);
	LOG4CXX_INFO(logger, "\tMean squared error targeted: " << this->ann_mse_target);

	return 1;
}

const std::string CliParser::get_input_image() const
{
	return this->input_image;
}

const std::vector<std::string> CliParser::get_class_images() const
{
	return this->class_images;
}

const std::string CliParser::get_export_dir() const
{
	return this->export_dir;
}

const int CliParser::get_export_interval() const
{
	return this->export_interval;
}

const int CliParser::get_num_iter() const
{
	return this->num_iter;
}

const double CliParser::get_lambda1() const {
	return this->lambda1;
}

const double CliParser::get_lambda2() const {
	return this->lambda2;
}

const std::vector< int > CliParser::get_ann_hidden_layers() const {
	return this->ann_hidden_layers;
}

const float CliParser::get_ann_learning_rate() const {
	return this->ann_learning_rate;
}

const unsigned int CliParser::get_ann_max_epoch() const {
	return this->ann_max_epoch.value;
}

const float CliParser::get_ann_mse_target() const {
	return this->ann_mse_target;
}

