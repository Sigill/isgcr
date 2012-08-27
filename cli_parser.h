#ifndef _CLI_OPTIONS_H
#define _CLI_OPTIONS_H

#include <boost/program_options.hpp>
#include <vector>

namespace po = boost::program_options;

class CliParser
{
public:
	CliParser();
	int parse_argv(int argc, char ** argv);
	const std::string get_input_image() const;
	const std::vector<std::string> get_class_images() const;
	const std::string get_export_dir() const;

private:
	std::string input_image;
	std::vector< std::string > class_images;
	std::string export_dir;
};

#endif /* _CLI_OPTIONS_H */
