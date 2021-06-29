#!/bin/bash

repeat=1

## BGPQ_T 1M, 8M, 64M with random, ascend, desend data.
#for ((i=0;i<${repeat};i++)); do
    #for n in 1 8 64; do
        #for t in 0 1 2; do
            #./BGPQ_T 0 0 ${n} ${t}
        #done
    #done
#done

## BGPQ_B 1M, 8M, 64M with random, ascend, desend data.
#for ((i=0;i<${repeat};i++)); do
    #for n in 1 8 64; do
        #for t in 0 1 2; do
            #./BGPQ_B 0 0 ${n} ${t}
        #done
    #done
#done

# BGPQ_T with empty, 1M, 64M initial keys.
for ((i=0;i<${repeat};i++)); do
    for n in 0 1 8; do
        ./BGPQ_T 1 ${n} 64 0
    done
done

# BGPQ_B with empty, 1M, 64M initial keys.
for ((i=0;i<${repeat};i++)); do
    for n in 0 1 8; do
        ./BGPQ_B 1 ${n} 64 0
    done
done


