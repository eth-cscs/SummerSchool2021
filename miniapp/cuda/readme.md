Setting up the environment for building the miniapp:

```
# to build

module load daint-gpu
module swap PrgEnv-cray PrgEnv-gnu
module load gcc/9.3.0 cudatoolkit
module load PyExtensions/python3-CrayGNU-20.11
make all

# to run

srun ./main 128 128 100 0.01

# to plot

python3 ./plotting.py -s
```

If you have an interactive session, you can uncomment the line `srun ./unit_tests`, which will run the unit tests every time you compile.
