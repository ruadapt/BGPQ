#!/bin/bash

repeat=1
#nthreads=40

# Figure 6(a), 6(b)
bnum=16
for ((i=0;i<${repeat};i++)); do
    for k in 128 256 512 1024; do
        for ((bsize=128;bsize<=${k};bsize*=2)); do
            ./BGPQ_T ${k} 27 ${bnum} ${bsize}
        done
    done
done

# Figure 6(c)
for ((i=0;i<${repeat};i++)); do
    for ((bnum=2;bnum<=128;bnum*=2)); do
        ./BGPQ_T 1024 27 ${bnum} 512
    done
done

bnum=16
for ((i=0;i<${repeat};i++)); do
    for k in 128 256 512 1024; do
        for ((bsize=128;bsize<=${k};bsize*=2)); do
            ./BGPQ_Tr ${k} 27 ${bnum} ${bsize}
        done
    done
done


