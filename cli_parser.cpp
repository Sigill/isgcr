#include "cli_parser.h"

#include <boost/filesystem.hpp>
#include <iostream>

CliParser::CliParser()
{}

int CliParser::parse_argv(int argc, char ** argv)
{
	po::options_description desc("Command line parameters");
	desc.add_options()
		("help,h", "Produce help message")
		("input-image,i", po::value< std::string >(&(this->input_image)), "Input image")
		("class-image,c", po::value< std::vector< std::string > >(&(this->class_images)), "Defines a class to be learned from a binary image")
		("export-dir,E", po::value< std::string >(&(this->export_dir)), "Export directory")
		("export-interval,e", po::value< unsigned int >(&(this->export_interval)), "Export interval during regularization")
		("num-iter,n", po::value< unsigned int >(&(this->num_iter)), "Number of iterations for the regularization")
		("lambda1", po::value< double >(&(this->lambda1)), "Lambda 1 parameter for regularization")
		("lambda2", po::value< double >(&(this->lambda2)), "Lambda 2 parameter for regularization")
		("num-gray", po::value< unsigned int >(&(this->num_gray)), "Number of gray levels used to compute the texture characteristics")
		("window_radius", po::value< unsigned int >(&(this->window_radius)), "Radius of the window used to compute the texture characteristics")
			;

	po::variables_map vm;
	po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
	po::notify(vm);

	if (vm.count("help")) {
		std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
		std::cout << desc;
		return 0;
	}

	if(vm.count("input-image"))
	{
		std::cout << "Image to process: " << this->input_image << std::endl;
		boost::filesystem::path path(this->input_image);
		if(!boost::filesystem::exists(path))
		{
			std::cerr << this->input_image << " does not exists" << std::endl;
			return -1;
		}
	} else {
		std::cerr << "No input image provided." << std::endl;
		return -1;
	}

	if(vm.count("class-image") && (this->class_images.size() >= 2))
	{
		std::cout << "Learning classes (" << this->class_images.size() << "): ";
		copy(this->class_images.begin(), this->class_images.end() - 1, std::ostream_iterator< std::string >(std::cout, ", ")); 
		std::cout << this->class_images.back() << std::endl;
	} else {
		std::cerr << "You should provide at least two learning classes." << std::endl;
		return -1;
	}

	if(vm.count("export-dir"))
	{
		std::cout << "Export directory: " << this->export_dir << std::endl;

		boost::filesystem::path path(this->export_dir);

		if(boost::filesystem::exists(path)) {
			if(boost::filesystem::is_directory(path)) {
				if(!boost::filesystem::is_empty(path)) {
					std::cerr << this->export_dir << " exists but is not empty" << std::endl;
					return -1;
				}
			} else {
				std::cerr << this->export_dir << " already exists as a file" << std::endl;
				return -1;
			}
		} else {
			if(!boost::filesystem::create_directories(path)) {
				std::cerr << this->export_dir << " cannot be created" << std::endl;
				return -1;
			}
		}
	} else {
		std::cerr << "No export directory provided." << std::endl;
		return -1;
	}

	if(!vm.count("export-interval"))
	{
		this->export_interval = 0;
	}
	std::cout << "Export interval during regularization: " << this->export_interval << std::endl;

	if(!vm.count("num-iter"))
	{
		this->num_iter = 0;
	}
	std::cout << "Number of iterations for regularization: " << this->num_iter << std::endl;

	if(!vm.count("lambda1"))
	{
		this->lambda1 = 1.0;
	}
	std::cout << "Lambda1 parameter for regularization: " << this->lambda1 << std::endl;

	if(!vm.count("lambda2"))
	{
		this->lambda2 = 1.0;
	}
	std::cout << "Lambda2 parameter for regularization: " << this->lambda2 << std::endl;

	if(!vm.count("num-gray"))
	{
		this->num_gray = 16;
	}
	std::cout << "Number of gray levels: " << this->num_gray << std::endl;

	if(!vm.count("window-radius"))
	{
		this->window_radius = 5;
	}
	std::cout << "Radius of the window: " << this->window_radius << std::endl;

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
