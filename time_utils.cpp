#include "time_utils.h"

#include <cstdlib>
#include <sys/time.h>

timestamp_t get_timestamp ()
{
  struct timeval now;
  gettimeofday (&now, NULL);
  return  now.tv_usec + (timestamp_t)now.tv_sec * 1000000;
}

float elapsed_time(timestamp_t t1, timestamp_t t2)
{
  return (t2 - t1) / 1000000.0;
}

