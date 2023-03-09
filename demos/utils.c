#include "utils.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

bool utils_gettime(double *pTime)
{
#ifdef _WIN32
    LARGE_INTEGER frequency, ts;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&ts);

    *pTime = (double)ts.QuadPart / frequency.QuadPart;
    return true;
#else
    struct timespec ts;

#ifdef __APPLE__
    if (__builtin_available(macOS 10.12, iOS 10.0, tvOS 10.0, watchOS 3.0, *))
#else
    if (true)
#endif
    {
        if (clock_gettime(CLOCK_MONOTONIC, &ts) < 0)
            return false;
    } else {
        return false;
    }

    *pTime = ts.tv_sec + ts.tv_nsec * 1e-9;
    return true;
#endif
}
