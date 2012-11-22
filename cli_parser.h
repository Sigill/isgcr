#ifndef _CLI_OPTIONS_H
#define _CLI_OPTIONS_H

#include <boost/program_options.hpp>
#include <vector>
#include <iostream>

#include <boost/regex.hpp>

namespace po = boost::program_options;

template <typename TNumericType>
class NumericTypeWrapper
{
public :
	typedef TNumericType NumericType;

	NumericTypeWrapper() {}
	NumericTypeWrapper(const NumericType v) : value(v) {}

	NumericType value;

	inline operator NumericType() const { return value; }
};

class StrictlyPositiveInteger : public NumericTypeWrapper<unsigned int> {
public:
	StrictlyPositiveInteger() : NumericTypeWrapper< unsigned int >() {}
	StrictlyPositiveInteger(const unsigned int v) : NumericTypeWrapper< unsigned int >(v) {}
};

class PositiveInteger : public NumericTypeWrapper<unsigned int> {
public:
	PositiveInteger() : NumericTypeWrapper< unsigned int >() {}
	PositiveInteger(const unsigned int v) : NumericTypeWrapper< unsigned int >(v) {}
};

class Float : public NumericTypeWrapper<float> {
public:
	Float() : NumericTypeWrapper< float >() {}
	Float(const float v) : NumericTypeWrapper< float >(v) {}
};

class Double : public NumericTypeWrapper<double> {
public:
	Double() : NumericTypeWrapper< double >() {}
	Double(const double v) : NumericTypeWrapper< double >(v) {}
};

/*
class StrictlyPositiveInteger {
public :
	StrictlyPositiveInteger() {}
	StrictlyPositiveInteger(const unsigned int v) : value(v) {}

	unsigned int value;

	inline operator unsigned int() const { return value; }
};

class PositiveInteger {
public :
	PositiveInteger() {}
	PositiveInteger(const unsigned int v) : value(v) {}

	unsigned int value;

	inline operator unsigned int() const { return value; }
};

class Float {
public :
	Float() {}
	Float(const float v) : value(v) {}

	float value;

	inline operator float() const { return value; }
};
*/
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
	const std::vector< unsigned int > get_ann_hidden_layers() const;
	const float get_ann_learning_rate() const;
	const unsigned int get_ann_max_epoch() const;
	const float get_ann_mse_target() const;

private:
	std::string input_image;
	std::vector< std::string > class_images;
	std::string export_dir;
	PositiveInteger export_interval;
	PositiveInteger num_iter;
	Double lambda1;
	Double lambda2;
	std::vector< unsigned int > ann_hidden_layers;
	Float ann_learning_rate;
	StrictlyPositiveInteger ann_max_epoch;
	Float ann_mse_target;
};

#endif /* _CLI_OPTIONS_H */
