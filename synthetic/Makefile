nvcc = nvcc
nvccflag = -std=c++11 -O3 -arch=sm_61 -DHEAP_SORT
heappath = ../heap/

all: BGPQ_T BGPQ_B

BGPQ_T: sort.cu $(heappath)tditdd_np/heap.cuh
	$(nvcc) $(nvccflag) $(testflag) -I$(heappath)tditdd_np/ $< -o $@

BGPQ_B: sort.cu $(heappath)buitdd_np/heap.cuh
	$(nvcc) $(nvccflag) $(testflag) -I$(heappath)buitdd_np/ $< -o $@

clean:
	rm -rf BGPQ_T BGPQ_B
