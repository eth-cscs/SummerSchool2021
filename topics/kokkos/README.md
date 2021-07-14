# Kokkos (Friday 23.07.2021)

This README contains pointers to the materials used in the summer school.

We will be using material prepared by the Kokkos team. For posterity, we will be
using commit `a9db29a9f24171c3919ae0bf62596cd9bd04fac9` of the [kokkos-tutorials
repository](https://github.com/kokkos/kokkos-tutorials) ([commit on
GitHub](https://github.com/kokkos/kokkos-tutorials/commit/a9db29a9f24171c3919ae0bf62596cd9bd04fac9)).

## Slides

We will primarily be using the ["Short" tutorial
material](https://github.com/kokkos/kokkos-tutorials/blob/a9db29a9f24171c3919ae0bf62596cd9bd04fac9/Intro-Short/KokkosTutorial_Short.pdf),
with selected material from the ["Medium"
tutorial](https://github.com/kokkos/kokkos-tutorials/blob/a9db29a9f24171c3919ae0bf62596cd9bd04fac9/Intro-Medium/KokkosTutorial_Medium.pdf)
subject to time.

## Exercises

The slides have pointers to appropriate exercises. All exercises can be found in
the [Exercises
subdirectory](https://github.com/kokkos/kokkos-tutorials/tree/a9db29a9f24171c3919ae0bf62596cd9bd04fac9/Exercises).

You're free to use whatever setup you want for building the exercises. However,
the following instructions are the recommended setup on Piz Daint.

Create a directory for Kokkos and the tutorial material:

``` sh
mkdir -p ~/Kokkos
```

Clone the Kokkos and tutorials repositories:

``` sh
git clone https://github.com/kokkos/kokkos.git ~/Kokkos/kokkos
git clone https://github.com/kokkos/kokkos-tutorials.git ~/Kokkos/kokkos-tutorials
```

Load the correct modules (we will be using GCC 9.3 with CUDA 11.0 since the
exercises assume the use of `g++` and `nvcc`; however, other combinations are
also possible):

``` sh
module load daint-gpu
module load cudatoolkit
module switch PrgEnv-cray/6.0.9 PrgEnv-gnu
module switch gcc/10.1.0 gcc/9.3.0
```

All exercises have a `Begin` and a `Solution` directory. The `Begin` directory
contains an exercise which requires some modification. The parts which need to
be changed or added in are marked with `EXERCISE` to make finding them easier.
The `Solution` directory contains a working solution.

All exercises come with a Makefile which is set up to use and build Kokkos
automatically from the cloned repository (Kokkos also comes with CMake support).
To build e.g. the first exercise, first go to the directory and then build it,
specifying the architecture that we want to build for (Haswell and P100 on Piz
Daint):

``` sh
cd ~/Kokkos/kokkos-tutorials/Exercises/01/Begin
make KOKKOS_ARCH=HSW,Pascal60 -j
```

The exercises have appropriate defaults for which backends ("devices") to use.
To change the default, you can specify the `KOKKOS_DEVICES` option. To
explicitly build with the OpenMP and CUDA backends, set it like this:

``` sh
make KOKKOS_ARCH=HSW,Pascal60 KOKKOS_DEVICES=OpenMP,Cuda -j
```

Finally, get an allocation on the GPU partition and run the exercise or
solution. Binaries compiled for the host only have a `.host` extension, while
binaries compiled for CUDA have a `.cuda` extension.

``` sh
salloc -A class07 -C gpu
srun ./01_Exercise.host
```
