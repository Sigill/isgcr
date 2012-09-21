#include "ConsolePluginProgress.h"
#include <iostream>
#include <iomanip>

ConsolePluginProgress::ConsolePluginProgress()
	: tlp::SimplePluginProgress(), comment("") {}

void ConsolePluginProgress::setComment(std::string comment)
{
	tlp::SimplePluginProgress::setComment(comment);

	std::cerr << comment << std::endl;
}

tlp::ProgressState ConsolePluginProgress::progress(int step, int max_step)
{
	std::cerr << "\r" << std::setfill(' ') << std::setw(3) << (int)(step / (float)max_step) << "%";

	return tlp::SimplePluginProgress::progress(step, max_step);
}
