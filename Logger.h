#ifndef LOGGER_H
#define LOGGER_H

#include <log4cxx/logger.h>

class Logger {
	public:
		static log4cxx::LoggerPtr getInstance();

	private:
		static log4cxx::LoggerPtr m_Instance;
};

#endif /* LOGGER_H */
