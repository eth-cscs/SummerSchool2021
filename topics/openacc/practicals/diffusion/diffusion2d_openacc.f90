program main
  use util
  use diffusion2d
  implicit none

  integer :: pow, nx, ny, nsteps, err, buffer_size, i
  real(kind(0d0)) :: dt, start_diffusion, time_diffusion
  real(kind(0d0)), dimension(:), allocatable :: x0, x1

  ! read arguments
  pow = read_arg(1, 8)
  nsteps = read_arg(2, 100)
  nx = 128+2
  ny = 2**pow + 2
  dt = 0.1

  write(*, *) ''
  write(*, '(a i0 a i0 a i0 a i0 a)') '## ', nx, 'x', ny, ' for ', nsteps, &
       ' time steps (', nx*ny, ' grid points)'

  buffer_size = nx*ny;
  allocate(x0(buffer_size), x1(buffer_size), stat=err)
  if (err /= 0) then
     stop 'failed to allocate arrays'
  endif

  ! TODO: Move data to the GPU

  ! set initial conditions of 0 everywhere
  call fill_gpu(x0, 0d0, buffer_size)
  call fill_gpu(x1, 0d0, buffer_size)

  ! set boundary conditions of 1 on south border
  call fill_gpu(x0, 1d0, nx);
  call fill_gpu(x1, 1d0, nx);
  call fill_gpu(x0(nx*(ny-1)+1:nx*ny), 1d0, nx);
  call fill_gpu(x1(nx*(ny-1)+1:nx*ny), 1d0, nx);

  !$acc wait
  start_diffusion = get_time()
  do i = 1, nsteps
     call diffusion_gpu(x0, x1, nx-2, ny-2, dt)
     call copy_gpu(x0, x1, buffer_size)
  enddo

  !$acc wait
  time_diffusion = get_time() - start_diffusion

  write(*, '(a f0.6 a e0.6e2 a)') '## ', time_diffusion, 's, ', &
       real(nsteps)*(nx-2)*(ny-2) / time_diffusion, ' points/second'
  write(*, *) ''
  write(*, *) ''
  write(*, '(a)') 'writing to output.bin/bov';
  write(*, *) ''

  call write_to_file(nx, ny, x1);

end program main
