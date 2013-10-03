#include "boost_program_options_types.h"
#include "ParseUtils.h"

namespace po = boost::program_options;

void validate(boost::any& v, const std::vector<std::string>& values, PositiveInteger*, int)
{
	po::validators::check_first_occurrence(v);
	const std::string& s = po::validators::get_single_string(values);

	unsigned int value;
	if( ParseUtils::ParseUInt(value, s.data(), 10) && value >= 0 )
	{
		v = boost::any(PositiveInteger(value));
	} else {
		throw po::invalid_option_value(s);
	}
}

void validate(boost::any& v, const std::vector<std::string>& values, StrictlyPositiveInteger*, int)
{
	po::validators::check_first_occurrence(v);
	const std::string& s = po::validators::get_single_string(values);

	unsigned int value;
	if( ParseUtils::ParseUInt(value, s.data(), 10) && value > 0 )
	{
		v = boost::any(StrictlyPositiveInteger(value));
	} else {
		throw po::invalid_option_value(s);
	}
}

void validate(boost::any& v, const std::vector<std::string>& values, Float*, int)
{
	po::validators::check_first_occurrence(v);
	const std::string& s = po::validators::get_single_string(values);

	float value;
	if( ParseUtils::ParseFloat(value, s.data()) )
	{
		v = boost::any(Float(value));
	} else {
		throw po::invalid_option_value(s);
	}
}

void validate(boost::any& v, const std::vector<std::string>& values, Double*, int)
{
	po::validators::check_first_occurrence(v);
	const std::string& s = po::validators::get_single_string(values);

	double value;
	if( ParseUtils::ParseDouble(value, s.data()) )
	{
		v = boost::any(Double(value));
	} else {
		throw po::invalid_option_value(s);
	}
}

void validate(boost::any& v, const std::vector<std::string>& values, Percentage*, int)
{
	po::validators::check_first_occurrence(v);
	const std::string& s = po::validators::get_single_string(values);

	float value;
	if( ParseUtils::ParseFloat(value, s.data()) && (value >= 0.0f) && (value <= 1.0f) )
	{
		v = boost::any(Percentage(value));
	} else {
		throw po::invalid_option_value(s);
	}
}
