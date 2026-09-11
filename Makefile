# QuBLAR -- Phase 1 build.
#
# nvcc is invoked from WSL on this machine: the Windows host has a link.exe collision in
# Git Bash, and the alternative (loading vcvars64 first) buys nothing here because none of
# Phase 1 links against a Windows library.

NVCC    ?= nvcc
ARCH    ?= sm_120
CXXFLAGS = -O3 -arch=$(ARCH) -lineinfo
SRC      = src
BUILD    = build

BINARIES = $(BUILD)/trace $(BUILD)/check_lidar

all: $(BINARIES)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/trace: $(SRC)/trace.cu $(SRC)/trace.cuh $(SRC)/bvh.hpp | $(BUILD)
	$(NVCC) $(CXXFLAGS) -o $@ $<

$(BUILD)/check_lidar: $(SRC)/check_lidar.cu $(SRC)/lidar.cuh $(SRC)/trace.cuh $(SRC)/bvh.hpp | $(BUILD)
	$(NVCC) $(CXXFLAGS) -o $@ $<

# Every invariant, both layers. Exit non-zero if any of them fails, so this is usable
# from a hook or a CI step without reading the output.
check: $(BINARIES)
	@$(BUILD)/trace && $(BUILD)/check_lidar

# Races and out-of-bounds accesses in the shared-memory accumulation, which no invariant
# above can see: a torn waveform bin still looks like a plausible waveform.
sanitize: $(BINARIES)
	compute-sanitizer --tool memcheck $(BUILD)/check_lidar
	compute-sanitizer --tool racecheck $(BUILD)/check_lidar

clean:
	rm -rf $(BUILD)

.PHONY: all check sanitize clean
