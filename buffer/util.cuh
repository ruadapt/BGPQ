#ifndef UTIL_CUH
#define UTIL_CUH

#include <numeric>
#include <algorithm>
#include <sys/time.h>

using namespace std;

void setTime(struct timeval* StartingTime) {
	gettimeofday(StartingTime, NULL);
}

double getTime(struct timeval* StartingTime, struct timeval* EndingingTime) {
	struct timeval ElapsedTime;
	timersub(EndingingTime, StartingTime, &ElapsedTime);
	return (ElapsedTime.tv_sec*1000.0 + ElapsedTime.tv_usec / 1000.0);	// Returning in milliseconds.
}

#endif
