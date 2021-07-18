Setting up the environment for building the miniapp:

```
# to build

module load daint-gpu
module swap PrgEnv-cray PrgEnv-gnu
module load cudatoolkit
make -j4

# to run

srun -n srun ./main 128 128 100 0.01

# to plot

../../scripts/plot.sh
```

If you have an interactive session, you can uncomment the line `srun ./unit_tests`, which will run the unit tests every time you compile.
