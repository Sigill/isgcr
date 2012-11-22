#include "ParseUtils.h"

#include <cerrno>
#include <cstdlib>
#include <climits>
#include <cmath>

// Inspired by: http://stackoverflow.com/questions/194465/how-to-parse-a-string-to-an-int-in-c

bool ParseUtils::ParseInt(int &value, char const *s, const int base)
{
	char *end;
	long v;
	errno = 0;
	v = strtol(s, &end, base);
	if ((errno == ERANGE && v == LONG_MAX) || v > INT_MAX || v < INT_MIN) {
		return false;
	}
	if (*s == '\0' || *end != '\0') {
		return false;
	}
	value = v;
	return true;
}

bool ParseUtils::ParseUInt(unsigned int &value, char const *s, const int base)
{
	char *end;
	unsigned long v;
	errno = 0;
	v = strtoul(s, &end, base);
	if ((errno == ERANGE && v == ULONG_MAX) || v > UINT_MAX) {
		return false;
	}
	if (*s == '\0' || *end != '\0') {
		return false;
	}
	value = v;
	return true;
}

bool ParseUtils::ParseFloat(float &value, char const *s)
{
	char *end;
	float v;
	errno = 0;
	v = strtof(s, &end);
	if (errno == ERANGE) {
		return false;
	}
	if (*s == '\0' || *end != '\0') {
		return false;
	}
	if(!std::isfinite(v)) {
		return false;
	}
	value = v;
	return true;
}

bool ParseUtils::ParseDouble(double &value, char const *s)
{
	char *end;
	double v;
	errno = 0;
	v = strtod(s, &end);
	if (errno == ERANGE) {
		return false;
	}
	if (*s == '\0' || *end != '\0') {
		return false;
	}
	if(!std::isfinite(v)) {
		return false;
	}
	value = v;
	return true;
}
