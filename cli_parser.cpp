#include "cli_parser.h"

#include <boost/filesystem.hpp>
#include <iostream>
#include <vector>

#include <boost/regex.hpp>

#include "log4cxx/logger.h"

namespace po = boost::program_options;

class cli_offset {
public :
	cli_offset(unsigned int o1, unsigned int o2, unsigned int o3) : m_offset(3)
	{
		m_offset[0] = o1;
		m_offset[1] = o2;
		m_offset[2] = o3;
	}

	std::vector< unsigned int > getOffset() { return m_offset; }

private:
	std::vector< unsigned int > m_offset;
};

void validate(boost::any& v, const std::vector<std::string>& values, cli_offset* target_type, int)
{
	static boost::regex r("(\\d+),(\\d+),(\\d+)");

	using namespace boost::program_options;

	// Make sure no previous assignment to 'v' was made.
	validators::check_first_occurrence(v);
	// Extract the first string from 'values'. If there is more than
	// one string, it's an error, and exception will be thrown.
	const std::string& s = validators::get_single_string(values);

	// Do regex match and convert the interesting part to int.
	boost::smatch match;
	if (boost::regex_match(s, match, r)) {
		v = boost::any(cli_offset(boost::lexical_cast<unsigned int>(match[1]), boost::lexical_cast<unsigned int>(match[2]), boost::lexical_cast<unsigned int>(match[3])));
	} else {
		throw invalid_option_value(s);
	}
}

std::ostream &operator<<(std::ostream &out, cli_offset& t){
	std::vector<unsigned int> vec = t.getOffset();

	std::copy(vec.begin(), vec.end(), std::ostream_iterator<unsigned int>(out, ", ") );

	return out;
}

CliParser::CliParser()
{}

int CliParser::parse_argv(int argc, char ** argv)
{
	log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("main"));

	LOG4CXX_INFO(logger, "Parsing command line options");

	cli_offset _offset(0, 0, 0);

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
		("num-gray",
			po::value< unsigned int >(&(this->num_gray))->default_value(16),
			"Number of gray levels used to compute the texture characteristics")
		( "window-radius",
			 po::value< unsigned int >(&(this->window_radius))->default_value(5),
			 "Radius of the window used to compute the texture characteristics")
		("offset",
			po::value< cli_offset >(&_offset)->required(),
			"Offset used to compute the texture characteristics")
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
	LOG4CXX_INFO(logger, "Number of gray levels: " << this->num_gray);
	LOG4CXX_INFO(logger, "Radius of the window: " << this->window_radius);
	{
		std::ostringstream m;
		m << _offset;
		LOG4CXX_INFO(logger, "Offset: " << m.str());
		this->offset = std::vector< unsigned int >(_offset.getOffset());
	}

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

const unsigned int CliParser::get_num_gray() const {
	return this->num_gray;
}

const unsigned int CliParser::get_window_radius() const {
	return this->window_radius;
}

const std::vector< unsigned int > CliParser::get_offset() const {
	return this->offset;
}
