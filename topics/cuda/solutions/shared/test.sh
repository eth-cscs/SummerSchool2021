for method in 0 1
do
    echo "... method $method"
    for n in `seq 10 2 26`
    do
        srun ./blur $n 200 $method | grep seconds
    done
    echo
done
