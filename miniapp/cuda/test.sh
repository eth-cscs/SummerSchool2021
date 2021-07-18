#!/bin/bash

for dim in 32 64 128 256 512 1024
do
    printf "%8d: " $dim
    srun -n1 ./main $dim $dim 10 0.001 | grep "rate of"
done
