# Serial miniapp implementation

To compile and run the serial version of the miniapp

```
# to build
module load daint-gpu
module swap PrgEnv-cray PrgEnv-gnu
make -j4

# to run without an allocation
srun -Cgpu --reservation=course ./main 128 128 128 0.01

# to run with an interactive session
salloc -Cgpu --reservation=course
srun -Cgpu ./main 128 128 128 0.01

# to plot
module load PyExtensions/2.7-CrayGNU-17.08
python2 ../plotting.py
```

Benchmark results on Piz Daint multicore partition with `srun main 256 256 100 0.01`

Measured in CG iterations/second.

```
           -------------------------
           | cray     gnu    intel |
------------------------------------
| C++      | 2129    1768     2201 |
------------------------------------

```
