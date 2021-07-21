export MPICH_RMA_OVER_DMAPP=1
export MPICH_RMA_USE_NETWORK_AMO=1

printf ".................................................\n"
printf "%25s%8s\n" "" "ranks"
printf "%9s%8s%8s%8s%8s%8s\n" "dim" 1 2 4 8 16
printf ".................................................\n"
for n in 128 256 512 1024 2048 4096 8192
do
    printf "%4dx%-4d" $n $n
    for r in 1 2 4 8 16
    do
        srun -n$r --ntasks-per-node=1 ./main $n $n 200 0.001 &> tmp
        iters=`grep "conjugate grad" tmp | awk '{print $8}'`
        printf "%8.0f" $iters
    done
    printf "\n"
done
printf ".................................................\n"

