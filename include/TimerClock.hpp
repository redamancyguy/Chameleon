//
// Created by wenli on 2022/12/29.

//

#ifndef TimerClock_hpp
#define TimerClock_hpp

#include <iostream>
#include <chrono>

class TimerClock {
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> _ticker;
public:
    TimerClock() {
        synchronization();
    }

    ~TimerClock() = default;

    void synchronization() {
        _ticker = std::chrono::high_resolution_clock::now();
    }

    double get_timer_second() {
        return (double) get_timer_nanoSec() * 1e-9;
    }

    double get_timer_milliSec() {
        return (double) get_timer_nanoSec() * 1e-6;
    }

    double get_timer_microSec() {
        return double(get_timer_nanoSec()) * 1e-3;
    }
    long long get_timer_nanoSec() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now() - _ticker).count();
    }

};


#endif //TimerClock_hpp
