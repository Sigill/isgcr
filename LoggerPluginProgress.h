#ifndef LOGGERPLUGINPROGRESS_H
#define LOGGERPLUGINPROGRESS_H

#include <string>
#include <tulip/SimplePluginProgress.h>
#include "log4cxx/logger.h"

class LoggerPluginProgress : public tlp::SimplePluginProgress {
public:
  LoggerPluginProgress(std::string logger_name);
  tlp::ProgressState progress(int step, int max_step);
  void setComment(std::string);

private:
  std::string comment;
  log4cxx::LoggerPtr logger;
};

#endif /* LOGGERPLUGINPROGRESS_H */
