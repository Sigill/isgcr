#ifndef PARSEUTILS_H
#define PARSEUTILS_H

class ParseUtils
{
public:
	enum ParseError { Success, Overflow, Inconvertible };

	static ParseUtils::ParseError ParseInt(int &i, char const *s, const int base = 0);
	static ParseUtils::ParseError ParseUInt(unsigned int &i, char const *s, const int base = 0);
};

#endif /* PARSEUTILS_H */
