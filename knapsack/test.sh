#!/bin/bash


for f in datasets/*.data; do
    ./knapsackT $f
done

for f in datasets/*.data; do
    ./knapsackB $f
done


