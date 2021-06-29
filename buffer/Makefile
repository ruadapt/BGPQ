NVCC=nvcc
NVCCFLAG= -arch=sm_61 -std=c++11
DEBUGFLAG= -g -G -arch=sm_61 -O0 -std=c++11 -DPRINT_DEBUG

all: bufferTest

bufferTest: bufferTest.cu buffer.cuh
	${NVCC} ${NVCCFLAG} $< -o $@

bufferTest_debug: bufferTest.cu buffer.cuh
	${NVCC} ${DEBUGFLAG} $< -o $@

clean:
	rm -rf bufferTest bufferTest_debug
