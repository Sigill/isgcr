#ifndef _TIME_UTILS_H
#define _TIME_UTILS_H

typedef unsigned long long timestamp_t;

timestamp_t get_timestamp();
float elapsed_time(timestamp_t t1, timestamp_t t2);

#endif /* _TIME_UTILS_H */

