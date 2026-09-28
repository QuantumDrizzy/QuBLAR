#pragma once
// A check speaks in the first instant, then again when it finishes.
// stdout is unbuffered: a pipe or a log file does not hold the line until exit.
// gpu_s is time inside the muon expose (device, synchronized).
// cpu_s is time inside the host anneal / local field.
// wall_s is the whole process. The rest is setup.
// dram_GBs is not on this line: these checks do not count device bytes.

#include <chrono>
#include <cstdio>

struct RunTalk {
    using clock = std::chrono::steady_clock;

    const char* name = "";
    clock::time_point wall0{};
    double gpu_s = 0;
    double cpu_s = 0;

    static RunTalk begin(const char* check_name) {
        std::setvbuf(stdout, nullptr, _IONBF, 0);
        RunTalk t;
        t.name = check_name;
        t.wall0 = clock::now();
        std::printf("QuBLAR %s start\n", check_name);
        std::fflush(stdout);
        return t;
    }

    void add(bool gpu, double seconds) {
        if (gpu) gpu_s += seconds;
        else cpu_s += seconds;
    }

    void end(const char* verdict) const {
        const double wall = std::chrono::duration<double>(clock::now() - wall0).count();
        std::printf(
            "QuBLAR %s done wall_s=%.3f gpu_s=%.3f cpu_s=%.3f dram_GBs=not_counted verdict=%s\n",
            name, wall, gpu_s, cpu_s, verdict);
        std::fflush(stdout);
    }
};

struct Tick {
    RunTalk* log = nullptr;
    bool gpu = false;
    RunTalk::clock::time_point t0{};

    Tick(RunTalk& talk, bool is_gpu) : log(&talk), gpu(is_gpu), t0(RunTalk::clock::now()) {}
    ~Tick() {
        const double s = std::chrono::duration<double>(RunTalk::clock::now() - t0).count();
        log->add(gpu, s);
    }
};
