#ifndef CONSOLEPLUGINPROGRESS_H
#define CONSOLEPLUGINPROGRESS_H

#include <string>
#include <tulip/SimplePluginProgress.h>

class ConsolePluginProgress : public tlp::SimplePluginProgress {
public:
  ConsolePluginProgress();
  tlp::ProgressState progress(int step, int max_step);
  void setComment(std::string);

private:
  std::string comment;
};

#endif /* CONSOLEPLUGINPROGRESS_H */
