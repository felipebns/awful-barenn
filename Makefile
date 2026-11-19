# -ffp-contract=off / -fmad=false: sem FMA, para a saída bater bit a bit com a do legado
CXX       = mpic++
OPT      ?= -O3
CXXFLAGS  = -std=c++17 $(OPT) -ffp-contract=off -Wall -Wextra

# rpath: nos nós do Franky o LD_LIBRARY_PATH às vezes perde a libstdc++ do g++ 14
LIBDIR   := $(dir $(shell g++ -print-file-name=libstdc++.so))
LDFLAGS   = -Wl,-rpath,$(LIBDIR)

NVCC       = nvcc
# RTX A1000 do Franky
ARCH      ?= -arch=sm_86
# ex.: DEFS=-DNSTREAMS=1
DEFS      ?=
NVCCFLAGS  = -std=c++17 -O3 -fmad=false $(ARCH) $(DEFS) -Xcompiler -ffp-contract=off

barenn: barenn.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

barenn_gpu: barenn.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@ -Xlinker -rpath,$(LIBDIR)

clean:
	rm -f barenn barenn_gpu

.PHONY: clean
