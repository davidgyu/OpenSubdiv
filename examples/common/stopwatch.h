//
//   Copyright 2013 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#ifndef STOPWATCH_H
#define STOPWATCH_H

#if (_WIN32)
    #include <windows.h>
#else
    #include <sys/types.h>
    #include <sys/time.h>
    #include <sys/resource.h>
#endif

class Stopwatch {

public:

#ifndef _WIN32
    Stopwatch() : _totalElapsed(0) { }

    void Start() {
        struct timeval l_rtime;
        gettimeofday(&l_rtime,0);
        _elapsed = (double)l_rtime.tv_sec + (double)l_rtime.tv_usec/1000000.0;
    }

    void Stop() {
        struct timeval l_rtime;

        gettimeofday(&l_rtime,0);
        _elapsed = ((double)l_rtime.tv_sec + (double)l_rtime.tv_usec/1000000.0)
                 - _elapsed;
        _totalElapsed += _elapsed;
    }

    double GetElapsed() const {
        return _elapsed;
    }

    double GetTotalElapsed() const {
        return _totalElapsed;
    }
#else
    Stopwatch() : _totalElapsed(0) {
        QueryPerformanceFrequency(&_frequency);
    }

    void Start()
    {
        QueryPerformanceCounter(&_time);
    }

    void Stop()
    {
        LARGE_INTEGER currentTime;
        QueryPerformanceCounter(&currentTime);
        _elapsed = currentTime.QuadPart - _time.QuadPart;
        _totalElapsed+=_elapsed;
    }

    double GetElapsed() const {
        return (double) _elapsed / _frequency.QuadPart;
    }

    double GetTotalElapsed() const {
        return (double) _totalElapsed / _frequency.QuadPart;
    }
#endif

private:

#ifndef _WIN32
    double _elapsed;
    double _totalElapsed;
#else
    LARGE_INTEGER _time;
    LARGE_INTEGER _frequency;
    __int64 _elapsed;
    __int64 _totalElapsed;
#endif

};

#endif /* STOPWATCH_H */

