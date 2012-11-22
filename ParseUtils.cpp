#include "ParseUtils.h"

#include <cerrno>
#include <cstdlib>
#include <climits>

ParseUtils::ParseError ParseUtils::ParseInt(int &i, char const *s, const int base)
{
	char *end;
	long  l;
	errno = 0;
	l = strtol(s, &end, base);
	if ((errno == ERANGE && l == LONG_MAX) || l > INT_MAX || l < INT_MIN) {
		return ParseUtils::Overflow;
	}
	if (*s == '\0' || *end != '\0') {
		return ParseUtils::Inconvertible;
	}
	i = l;
	return ParseUtils::Success;
}

ParseUtils::ParseError ParseUtils::ParseUInt(unsigned int &i, char const *s, const int base)
{
	char *end;
	unsigned long  l;
	errno = 0;
	l = strtoul(s, &end, base);
	if ((errno == ERANGE && l == ULONG_MAX) || l > UINT_MAX) {
		return ParseUtils::Overflow;
	}
	if (*s == '\0' || *end != '\0') {
		return ParseUtils::Inconvertible;
	}
	i = l;
	return ParseUtils::Success;
}
