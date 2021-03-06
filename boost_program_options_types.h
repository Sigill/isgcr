#ifndef BOOST_PROGRAM_OPTIONS_TYPES_H
#define BOOST_PROGRAM_OPTIONS_TYPES_H

#include <boost/program_options.hpp>

/**
 * Numeric type wrapper.
 */
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

/**
 * Strictly positive integer wrapper.
 *
 * No check is performed when creating the object.
 */
class StrictlyPositiveInteger : public NumericTypeWrapper<unsigned int> {
public:
	StrictlyPositiveInteger() : NumericTypeWrapper< unsigned int >() {}
	StrictlyPositiveInteger(const unsigned int v) : NumericTypeWrapper< unsigned int >(v) {}
};

/**
 * Positive integer wrapper.
 *
 * No check is performed when creating the object.
 */
class PositiveInteger : public NumericTypeWrapper<unsigned int> {
public:
	PositiveInteger() : NumericTypeWrapper< unsigned int >() {}
	PositiveInteger(const unsigned int v) : NumericTypeWrapper< unsigned int >(v) {}
};

/**
 * Float wrapper.
 *
 * No check is performed when creating the object.
 */
class Float : public NumericTypeWrapper<float> {
public:
	Float() : NumericTypeWrapper< float >() {}
	Float(const float v) : NumericTypeWrapper< float >(v) {}
};

/**
 * Double wrapper.
 *
 * No check is performed when creating the object.
 */
class Double : public NumericTypeWrapper<double> {
public:
	Double() : NumericTypeWrapper< double >() {}
	Double(const double v) : NumericTypeWrapper< double >(v) {}
};

/**
 * Percentage wrapper. This is actually a float.
 *
 * No check is performed when creating the object.
 */
class Percentage : public NumericTypeWrapper<float> {
public:
	Percentage() : NumericTypeWrapper< float >() {}
	Percentage(const float v) : NumericTypeWrapper< float >(v) {}
};

/**
 * Validator used by boost::program_options for PositiveInteger.
 *
 * \sa PositiveInteger.
 */
void validate(boost::any& v, const std::vector<std::string>& values, PositiveInteger*, int);

/**
 * Validator used by boost::program_options for StrictlyPositiveInteger.
 *
 * \sa StrictlyPositiveInteger.
 */
void validate(boost::any& v, const std::vector<std::string>& values, StrictlyPositiveInteger*, int);

/**
 * Validator used by boost::program_options for float.
 *
 * \sa Float.
 */
void validate(boost::any& v, const std::vector<std::string>& values, Float*, int);

/**
 * Validator used by boost::program_options for Double.
 *
 * \sa Double.
 */
void validate(boost::any& v, const std::vector<std::string>& values, Double*, int);

/**
 * Validator used by boost::program_options for Percentage.
 *
 * \sa Percentage.
 */
void validate(boost::any& v, const std::vector<std::string>& values, Percentage*, int);

#endif /* BOOST_PROGRAM_OPTIONS_TYPES_H */
