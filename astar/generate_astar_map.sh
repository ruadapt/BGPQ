#!/bin/bash

for size in 5000 10000 20000
do
    for rate in 20 30
    do
        echo $size
        ./generate_astar_map ${size} ${size} ${rate} ${FREESPACE}/astar_maps/${size}_${size}_${rate}.map
    done
done
