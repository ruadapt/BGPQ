nvcc = nvcc
nvccflag = -std=c++11 -O3 -arch=sm_61 -DHEAP_SORT #-DDEBUG #-DTHRUST_SORT #-DINSERT_SMEM
heappath = ../heap/

all: ssspT ssspB
	
ssspT: sssp.cu
	$(nvcc) $(nvccflag) -I$(heappath)tditdd_np/ $< -o $@

ssspB: sssp.cu
	$(nvcc) $(nvccflag) -I$(heappath)buitdd_np/ $< -o $@

clean:
	rm -rf ssspT ssspB
