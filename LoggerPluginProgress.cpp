#include "LoggerPluginProgress.h"
#include <iostream>
#include <iomanip>

LoggerPluginProgress::LoggerPluginProgress(std::string logger_name)
	: tlp::SimplePluginProgress(), comment("")
{
	this->logger = log4cxx::LoggerPtr(log4cxx::Logger::getLogger(logger_name));
}

void LoggerPluginProgress::setComment(std::string comment)
{
	tlp::SimplePluginProgress::setComment(comment);

	this->comment = comment;
}

tlp::ProgressState LoggerPluginProgress::progress(int step, int max_step)
{
	std::ostringstream status;
	status << this->comment << ": " << (int)(100 * step / (float)max_step) << "%";

	LOG4CXX_INFO(this->logger, status.str());

	return tlp::SimplePluginProgress::progress(step, max_step);
}

