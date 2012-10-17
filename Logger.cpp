#include "Logger.h"
#include "log4cxx/basicconfigurator.h"
#include "log4cxx/consoleappender.h"
#include "log4cxx/patternlayout.h"

// Global static pointer used to ensure a single instance of the class.
log4cxx::LoggerPtr Logger::m_Instance = log4cxx::LoggerPtr();

log4cxx::LoggerPtr Logger::getInstance()
{
	if (!m_Instance)   // Only allow one instance of class to be generated.
	{
		m_Instance = log4cxx::LoggerPtr(log4cxx::Logger::getLogger("main"));

		log4cxx::BasicConfigurator::configure(
				log4cxx::AppenderPtr(new log4cxx::ConsoleAppender(
							log4cxx::LayoutPtr(new log4cxx::PatternLayout("\%-5p - \%m\%n")),
							log4cxx::ConsoleAppender::getSystemErr()
						)
					)
				);
	}

	return m_Instance;
}

