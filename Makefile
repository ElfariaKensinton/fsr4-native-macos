# fsr4-mac build
CXX      := clang++
FSR4_MIN_MACOS ?= 26.0
CXXFLAGS := -std=c++17 -O2 -Wall -mmacosx-version-min=$(FSR4_MIN_MACOS)
SDKROOT  := $(shell xcrun --sdk macosx --show-sdk-path)
FRAMEWORKS := -framework Metal -framework Foundation
ZLIB     := -lz
BINDIR   := tools/bin

all: $(BINDIR)/fsr4mac $(BINDIR)/gen_testcase $(BINDIR)/compare $(BINDIR)/bilin $(BINDIR)/bin2png

test:
	mkdir -p build/tests
	$(CXX) $(CXXFLAGS) tests/backend_policy_test.cpp -o build/tests/backend_policy_test
	build/tests/backend_policy_test
	python3 -m unittest discover -s tests -p 'test_*.py'

test-gpu: $(BINDIR)/fsr4mac $(BINDIR)/gen_testcase
	python3 tools/bench/check_backends.py

$(BINDIR)/fsr4mac: Makefile src/fsr4mac.mm src/pngio.h src/pass0_weights.h src/i8mpp_host.h src/fsr4_backend.h src/fsr4_device.h
	$(CXX) $(CXXFLAGS) -isysroot $(SDKROOT) $(FRAMEWORKS) $(ZLIB) -o $@ src/fsr4mac.mm

$(BINDIR)/gen_testcase: src/gen_testcase.cpp
	$(CXX) $(CXXFLAGS) -o $@ src/gen_testcase.cpp

$(BINDIR)/compare: src/compare.cpp src/pngio.h
	$(CXX) $(CXXFLAGS) $(ZLIB) -o $@ src/compare.cpp

$(BINDIR)/bilin: src/bilin.cpp src/pngio.h
	$(CXX) $(CXXFLAGS) $(ZLIB) -o $@ src/bilin.cpp

$(BINDIR)/bin2png: src/bin2png.cpp src/pngio.h
	$(CXX) $(CXXFLAGS) $(ZLIB) -o $@ src/bin2png.cpp

clean:
	rm -f $(BINDIR)/fsr4mac $(BINDIR)/gen_testcase $(BINDIR)/compare $(BINDIR)/bilin $(BINDIR)/bin2png

.PHONY: all clean test test-gpu
