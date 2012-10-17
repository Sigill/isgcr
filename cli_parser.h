#ifndef _CLI_OPTIONS_H
#define _CLI_OPTIONS_H

#include <boost/program_options.hpp>
#include <vector>

class CliParser
{
public:
	CliParser();
	int parse_argv(int argc, char ** argv);
	const std::string get_input_image() const;
	const std::vector<std::string> get_class_images() const;
	const std::string get_export_dir() const;
	const unsigned int get_export_interval() const;
	const unsigned int get_num_iter() const;
	const double get_lambda1() const;
	const double get_lambda2() const;
	const unsigned int get_num_gray() const;
	const unsigned int get_window_radius() const;
	const std::vector< unsigned int > get_offset() const;

private:
	std::string input_image;
	std::vector< std::string > class_images;
	std::string export_dir;
	unsigned int export_interval;
	unsigned int num_iter;
	double lambda1;
	double lambda2;
	unsigned int window_radius;
	unsigned int num_gray;
	std::vector< unsigned int > offset;
};

#endif /* _CLI_OPTIONS_H */
