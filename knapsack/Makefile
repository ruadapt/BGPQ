NVCC=nvcc
NVCCFLAG= -arch=sm_61 -O3 -std=c++11
KNAPSACK_FLAG=-DPBS_MODEL
DEBUG_FLAG=-DPERF_DEBUG -DPRINT_DEBUG

TD_HEAP_DIR=../heap/tditdd/
BU_HEAP_DIR=../heap/buitdd/
BUFFER_DIR=../buffer/

all: knapsackT knapsackB

knapsackT: main.cu models.cuh models_fifo.cuh models_pq.cuh models_pq_fifo.cuh gc.cuh
	${NVCC} ${NVCCFLAG} -I${TD_HEAP_DIR} -I${BUFFER_DIR} ${KNAPSACK_FLAG} $< -o $@

knapsackT_debug: main.cu models.cuh models_fifo.cuh models_pq.cuh models_pq_fifo.cuh gc.cuh
	${NVCC} ${NVCCFLAG} -I${TD_HEAP_DIR} -I${BUFFER_DIR} ${KNAPSACK_FLAG} ${DEBUG_FLAG} $< -o $@

knapsackB: main.cu models.cuh models_fifo.cuh models_pq.cuh models_pq_fifo.cuh gc.cuh
	${NVCC} ${NVCCFLAG} -I${BU_HEAP_DIR} -I${BUFFER_DIR} ${KNAPSACK_FLAG} $< -o $@

knapsackB_debug: main.cu models.cuh models_fifo.cuh models_pq.cuh models_pq_fifo.cuh gc.cuh
	${NVCC} ${NVCCFLAG} -I${BU_HEAP_DIR} -I${BUFFER_DIR} ${KNAPSACK_FLAG} ${DEBUG_FLAG} $< -o $@


#knapsack_fifo: main.cu models_fifo.cuh
#	${NVCC} ${NVCCFLAG} -I${TD_HEAP_DIR} -I${BUFFER_DIR} ${KNAPSACK_FLAG} ${DEBUG_FLAG} $< -o $@

clean:
	rm -rf knapsackT knapsackB knapsackT_debug knapsackB_debug
