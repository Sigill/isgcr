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
		("export-dir,e", po::value< std::string >(&(this->export_dir)), "Export directory")
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

