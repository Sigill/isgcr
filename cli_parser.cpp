#include "cli_parser.h"

#include <boost/filesystem.hpp>
#include <iostream>
#include <vector>

#include <boost/regex.hpp>

#include "log4cxx/logger.h"

namespace po = boost::program_options;

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
			po::value< unsigned int >(&(this->export_interval))->default_value(0),
			"Export interval during regularization")
		("num-iter,n",
			po::value< unsigned int >(&(this->num_iter))->default_value(0),
			"Number of iterations for the regularization")
		("lambda1",
			po::value< double >(&(this->lambda1))->default_value(1.0),
			"Lambda 1 parameter for regularization")
		("lambda2",
			po::value< double >(&(this->lambda2))->default_value(1.0),
			"Lambda 2 parameter for regularization")
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

	LOG4CXX_INFO(logger, "Export directory: " << this->export_dir);
	LOG4CXX_INFO(logger, "Export interval during regularization: " << this->export_interval);
	LOG4CXX_INFO(logger, "Number of iterations for regularization: " << this->num_iter);
	LOG4CXX_INFO(logger, "Lambda2 parameter for regularization: " << this->lambda2);
	LOG4CXX_INFO(logger, "Lambda1 parameter for regularization: " << this->lambda1);

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

const unsigned int CliParser::get_export_interval() const
{
	return this->export_interval;
}

const unsigned int CliParser::get_num_iter() const
{
	return this->num_iter;
}

const double CliParser::get_lambda1() const {
	return this->lambda1;
}

const double CliParser::get_lambda2() const {
	return this->lambda2;
}
