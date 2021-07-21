Setting up the environment for building the miniapp:

```
#
# to build
#

module load daint-gpu
module swap PrgEnv-cray PrgEnv-gnu
module load gcc/9.3.0 cudatoolkit
make -j4

#
# to run
#

# required if using rdma
export MPICH_RDMA_ENABLED_CUDA=1
srun -Cgpu -n4 -N4 ./main 128 128 100 0.01 true

#
# to plot
#

module load PyExtensions
python3 ../../plotting.py
```

Note that you only seed speedup using more than one MPI rank for large problems.

Here is the throughput of the application, measured in iterations per second as
the number of mpi ranks and the problem size varies.

```
.................................................
                            ranks
      dim       1       2       4       8      16
.................................................
 128x128     5343    3243    2586    1902    1555
 256x256     5175    3636    2856    2126    1752
 512x512     4271    3521    2942    2241    1962
1024x1024    1982    2327    2635    2408    1897
2048x2048     661    1076    1552    1781    1688
4096x4096     146     277     506     776    1107
8192x8192      37      73     142     260     443
.................................................
```

Every time we double the dimension the problem is 4x larger, yet for 1 mpi rank, we
initially see about the same throughput for 128x128 and 256x256, where you might
expecte the throughput to be 4 times less for the 256 case.
The reason is that there is not enough work to utilize a single GPU.
So, for small problems we see that using more MPI ranks decreases throughput.
It isn't until 1024x1024 that we see any speedup for 2 ranks relative to 1.
And you need to got 4096x4096 to see sustained speedup all the way to 16 ranks.

Here are the same results without RDMA. There is some improvement using RDMA, particularly for
smaller problems with many ranks. For the largest problem on 16 ranks, it gives
a speedup of 1.12, which is pretty good!

```
.................................................
                            ranks
      dim       1       2       4       8      16
.................................................
 128x128     5352    3231    2396    1877    1448
 256x256     5131    3486    2719    2151    1547
 512x512     4249    3327    2805    2242    1834
1024x1024    1979    2287    2545    2245    1813
2048x2048     661    1053    1469    1597    1683
4096x4096     146     278     491     758     988
8192x8192      37      73     142     251     429
.................................................
```



