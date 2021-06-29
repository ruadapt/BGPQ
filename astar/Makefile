NVCCFLAG= -arch=sm_61 -O3 -std=c++11 #--ptxas-options=-v
NVCC=nvcc ${NVCCFLAG}

all: astarT astarB

astarT: astar_main.cu *.cuh *.hpp BGPQ_T/heap.cuh
	${NVCC} astar_main.cu -I BGPQ_T/ -o astarT

astarB: astar_main.cu *.cuh *.hpp BGPQ_B/heap.cuh
	${NVCC} astar_main.cu -I BGPQ_B/ -o astarB

astarT_debug: astar_main.cu *.cuh *.hpp BGPQ_T/heap.cuh
	${NVCC} -DDEBUG astar_main.cu -I BGPQ_T/ -o astarT

astarB_debug: astar_main.cu *.cuh *.hpp BGPQ_B/heap.cuh
	${NVCC} -DDEBUG astar_main.cu -I BGPQ_B/ -o astarB

generate_astar_map: generate_astar_map.cu astar_map.cuh
	${NVCC} generate_astar_map.cu -o generate_astar_map

clean:
	rm -rf astarT astarB generate_astar_map
	
