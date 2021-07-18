## CUDA Lesson Plan

### Day 1

#### Introduction

Understand motivations behind using GPUs for HPC. Key architectural distinctions between CPU & GPU.

`introduction.pdf`

#### CUDA API

Learn the programming model, common GPU libraries, understand GPU memory management with practical exercises. 

`porting.pdf`

`runtime_api.pdf`

`memory.pdf`

Exercises under `practicals/api` folder.

### Day 2

#### Kernels & Threads

Writing custom GPU kernels, understanding concepts of CUDA threads, blocks and grids with practical exercises  

`kernels.pdf`

Exercises under `practicals/axpy` folder.

#### Shared Memory and Block Syncronization

Learn using cooperating thread blocks for more advanced kernels. Understand concepts such as race conditions, thread synchronization, atomics with practical exercizes. 

`shared.pdf`

Exercises under `practicals/shared` folder.

### Day 3

#### 2D Diffusion Miniapp

Understand implementing a real-world numerical simulation using a toy mini-app. Leverage previous concepts to implement working GPU code, and compare with a CPU version. The same example would be extended for future lessons on OpenACC as well.

`miniapp_intro.pdf`

`miniapp.pdf`

`cuda2d.pdf`

Coding exercizes in the `miniapp` folder. Contains a working `OpenMP` implementation as well.

#### Advanced GPU Concepts

Asynchronous operations for concurrency, and using GPUs in distributed computing.

`async.pdf`

`cuda_mpi.pdf`

Exercises under `practicals/async` folder.

#### NOTE: Solutions would be uploaded in the end of the day in the same repo in the `solutions/` folder.