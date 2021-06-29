#!/bin/bash

repeat=1
#nthreads=40

for ((i=0;i<${repeat};i++)); do
    for s in 5000 10000 20000; do
        for o in 10 20; do
            ./astarT datasets/${s}_${s}_${o}.map
        done
    done
done

for ((i=0;i<${repeat};i++)); do
    for s in 5000 10000 20000; do
        for o in 10 20; do
            ./astarB datasets/${s}_${s}_${o}.map
        done
    done
done
