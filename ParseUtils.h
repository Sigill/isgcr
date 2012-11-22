#ifndef PARSEUTILS_H
#define PARSEUTILS_H

class ParseUtils
{
public:
	static bool ParseInt(int &i, char const *s, const int base = 0);
	static bool ParseUInt(unsigned int &i, char const *s, const int base = 0);
};

#endif /* PARSEUTILS_H */
