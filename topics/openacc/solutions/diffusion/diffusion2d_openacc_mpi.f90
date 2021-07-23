program main
  use mpi
  use util
  use diffusion2d
  implicit none

  integer :: pow, nx, ny, nsteps, err, buffer_size, i, mpi_rank, mpi_size, &
       south, north, num_requests
  real(kind(0d0)) :: dt, start_diffusion, time_diffusion
  real(kind(0d0)), dimension(:), allocatable :: x0, x1
  integer, dimension(4) :: requests
  integer, dimension(4) :: statuses(MPI_STATUS_SIZE)

  call mpi_init(err)
  call mpi_comm_rank(MPI_COMM_WORLD, mpi_rank, err)
  call mpi_comm_size(MPI_COMM_WORLD, mpi_size, err)

  ! read arguments
  pow = read_arg(1, 8)
  nsteps = read_arg(2, 100)

  ! set domain size
  nx = 128
  ny = 2**pow
  dt = 0.1

  if (mod(ny, mpi_size) /= 0) then
     write(*, '(a i0 a i0)') 'error : global domain dimension ', ny, &
          'must be divisible by number of MPI ranks ', mpi_size;
     stop
  else if (mpi_rank == 0) then
     write(*, *) ''
     write(*, '(a i0 a)') '##', mpi_size, ' MPI ranks'
     write(*, '(a i0 a i0 a i0 a i0 a i0 a i0 a i0)') &
          '## ', nx, 'x', ny, ' : ', nx, 'x', ny/mpi_size, ' per rank for ', &
          nsteps, ' time steps (', nx*ny, ' grid points)'
  endif

  ny = ny / mpi_size
  nx = nx + 2
  ny = ny + 2
  buffer_size = nx*ny

  allocate(x0(buffer_size), x1(buffer_size), stat=err)
  if (err /= 0) then
     stop 'failed to allocate arrays'
  endif

  !$acc data create(x0) copyout(x1)

  ! set initial conditions of 0 everywhere
  call fill_gpu(x0, 0d0, buffer_size)
  call fill_gpu(x1, 0d0, buffer_size)

  ! set boundary conditions of 1 on south border
  if (mpi_rank == 0) then
     call fill_gpu(x0, 1d0, nx);
     call fill_gpu(x1, 1d0, nx);
  endif

  if (mpi_rank == mpi_size-1) then
     call fill_gpu(x0(nx*(ny-1)+1:nx*ny), 1d0, nx);
     call fill_gpu(x1(nx*(ny-1)+1:nx*ny), 1d0, nx);
  endif


  south = mpi_rank - 1
  north = mpi_rank + 1

  requests = 0
  statuses = 0

  !$acc wait
  start_diffusion = get_time()
  do i = 1, nsteps
     num_requests = 0
     !$acc host_data use_device(x0, x1)
     if (south >= 0) then
        call mpi_irecv(x0(1:), nx, MPI_DOUBLE, south, 0, MPI_COMM_WORLD, &
             requests(1), err)
        call mpi_isend(x0(nx+1:), nx, MPI_DOUBLE, south, 0, MPI_COMM_WORLD, &
             requests(2), err)
        num_requests = num_requests + 2
     endif

     if (north < mpi_size) then
        call mpi_irecv(x0((ny-1)*nx+1:), nx, MPI_DOUBLE, north, 0, &
             MPI_COMM_WORLD, requests(num_requests+1), err)
        call mpi_isend(x0((ny-2)*nx+1:), nx, MPI_DOUBLE, north, 0, &
             MPI_COMM_WORLD, requests(num_requests+2), err)
        num_requests = num_requests + 2
     endif
     !$acc end host_data

     call mpi_waitall(num_requests, requests, statuses, err)

     call diffusion_gpu(x0, x1, nx-2, ny-2, dt)

     call copy_gpu(x0, x1, buffer_size)
  enddo

  !$acc wait
  time_diffusion = get_time() - start_diffusion

  !$acc end data

  if (mpi_rank == 0) then
     write(*, '(a f0.6 a e0.6e2 a)') '## ', time_diffusion, 's, ', &
          real(nsteps)*(nx-2)*(ny-2) / time_diffusion, ' points/second'
     write(*, *) ''
     write(*, *) ''
     write(*, '(a)') 'writing to output.bin/bov'
     write(*, *) ''
  endif

  call write_to_file(nx, ny, x1);
  call MPI_Finalize(err)

end program main
