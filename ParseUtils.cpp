#include "ParseUtils.h"

#include <cerrno>
#include <cstdlib>
#include <climits>

// Inspired by: http://stackoverflow.com/questions/194465/how-to-parse-a-string-to-an-int-in-c

bool ParseUtils::ParseInt(int &i, char const *s, const int base)
{
	char *end;
	long  l;
	errno = 0;
	l = strtol(s, &end, base);
	if ((errno == ERANGE && l == LONG_MAX) || l > INT_MAX || l < INT_MIN) {
		return false;
	}
	if (*s == '\0' || *end != '\0') {
		return false;
	}
	i = l;
	return true;
}

bool ParseUtils::ParseUInt(unsigned int &i, char const *s, const int base)
{
	char *end;
	unsigned long  l;
	errno = 0;
	l = strtoul(s, &end, base);
	if ((errno == ERANGE && l == ULONG_MAX) || l > UINT_MAX) {
		return false;
	}
	if (*s == '\0' || *end != '\0') {
		return false;
	}
	i = l;
	return true;
}
