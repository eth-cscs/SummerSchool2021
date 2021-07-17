
printf "%3s%16s%16s%10s\n" "n" "OpenMP" "CUDA" "speedup"
for n in `seq 10 10`
do
    OMP_NUM_THREADS=1 srun -n1 -c12 --hint=nomultithread ./axpy_omp $n > run_omp
    srun ./axpy $n > run_cuda
    time_omp=`grep ^axpy run_omp | awk '{printf("%16.10f", $2)}'`
    time_cuda=`grep ^axpy run_cuda | awk '{printf("%16.10f", $2)}'`
    speedup=`echo $time_omp/$time_cuda | bc -l`
    printf "%3d%16.10f%16.10f%16.10f\n" $n $time_omp $time_cuda $speedup
done
for n in `seq 16 1 29`
do
    srun -n1 -c12 --hint=nomultithread ./axpy_omp $n > run_omp
    srun ./axpy $n > run_cuda
    time_omp=`grep ^axpy run_omp | awk '{printf("%16.10f", $2)}'`
    time_cuda=`grep ^axpy run_cuda | awk '{printf("%16.10f", $2)}'`
    speedup=`echo $time_omp/$time_cuda | bc -l`
    printf "%3d%16.10f%16.10f%16.10f\n" $n $time_omp $time_cuda $speedup
done
