#!/bin/bash

repeat=1
#nthreads=40

for ((i=0;i<${repeat};i++)); do
    ./ssspT datasets/com-LiveJournal.out 
    ./ssspT datasets/mawi_201512020000.out 
    ./ssspT datasets/hollywood-2009.out

    ./ssspB datasets/com-LiveJournal.out 
    ./ssspB datasets/mawi_201512020000.out 
    ./ssspB datasets/hollywood-2009.out
done
