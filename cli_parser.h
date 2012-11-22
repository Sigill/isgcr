#ifndef _CLI_OPTIONS_H
#define _CLI_OPTIONS_H

#include <boost/program_options.hpp>
#include <vector>
#include <iostream>

#include <boost/regex.hpp>

namespace po = boost::program_options;

class strictly_positive_integer {
public :
	explicit strictly_positive_integer() : value(1) {}
	explicit strictly_positive_integer(const unsigned int v) : value(v) {}

	unsigned int value;
};

class positive_integer {
public :
	explicit positive_integer() : value(1) {}
	explicit positive_integer(const unsigned int v) : value(v) {}

	unsigned int value;
};

class CliParser
{
public:
	CliParser();
	int parse_argv(int argc, char ** argv);
	const std::string get_input_image() const;
	const std::vector<std::string> get_class_images() const;
	const std::string get_export_dir() const;
	const int get_export_interval() const;
	const int get_num_iter() const;
	const double get_lambda1() const;
	const double get_lambda2() const;
	const std::vector< int > get_ann_hidden_layers() const;
	const float get_ann_learning_rate() const;
	const unsigned int get_ann_max_epoch() const;
	const float get_ann_mse_target() const;

private:
	std::string input_image;
	std::vector< std::string > class_images;
	std::string export_dir;
	int export_interval;
	int num_iter;
	double lambda1;
	double lambda2;
	std::vector< int > ann_hidden_layers;
	float ann_learning_rate;
	strictly_positive_integer ann_max_epoch;
	float ann_mse_target;
};

#endif /* _CLI_OPTIONS_H */
