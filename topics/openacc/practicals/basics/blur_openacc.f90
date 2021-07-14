real(kind(0d0)) function blur(pos, u, n)
  ! TODO: Declare as OpenACC routine accordingly
  integer, intent(in) :: pos, n
  real(kind(0d0)), intent(in) :: u(n)

  blur = 0.25*(u(pos-1) + 2.0*u(pos) + u(pos+1))
end function blur

subroutine blur_twice_host(nsteps, n, in, out)
  use util
  implicit none
  integer, intent(in) :: n, nsteps
  real(kind(0d0)), intent(inout) :: in(n)
  real(kind(0d0)), intent(inout) :: out(n)

  integer istep, i
  real(kind(0d0)), dimension(:), allocatable :: buffer
  real(kind(0d0)), external :: blur

  allocate(buffer(n))

  do istep = 1,nsteps
     !$omp parallel do
     do i = 2,n-1
        buffer(i) = blur(i, in, n)
     enddo

     !$omp parallel do
     do i = 3,n-2
        out(i) = blur(i, buffer, n)
     enddo

     !$omp parallel do
     do i = 1,n
        in(i) = out(i)
     enddo
  enddo

  deallocate(buffer)
end subroutine blur_twice_host

subroutine blur_twice_gpu_naive(nsteps, n, in, out)
  use util
  implicit none
  integer, intent(in) :: n, nsteps
  real(kind(0d0)), intent(inout) :: in(n)
  real(kind(0d0)), intent(inout) :: out(n)

  integer istep, i
  real(kind(0d0)), dimension(:), allocatable :: buffer
  real(kind(0d0)), external :: blur
  ! TODO: Declare as OpenACC routine accordingly

  allocate(buffer(n))

  do istep = 1,nsteps
     ! TODO: Offload this loop to the GPU
     do i = 2,n-1
        buffer(i) = blur(i, in, n)
     enddo

     ! TODO: Offload this loop to the GPU
     do i = 3,n-2
        out(i) = blur(i, buffer, n)
     enddo

     ! TODO: Offload this loop to the GPU
     do i = 1,n
        in(i) = out(i)
     enddo
  enddo

  deallocate(buffer)
end subroutine blur_twice_gpu_naive

subroutine blur_twice_gpu_nocopies(nsteps, n, in, out)
  implicit none
  integer, intent(in) :: n, nsteps
  real(kind(0d0)), intent(inout) :: in(n)
  real(kind(0d0)), intent(inout) :: out(n)

  integer istep, i
  real(kind(0d0)), dimension(:), allocatable :: buffer
  real(kind(0d0)), external :: blur
  ! TODO: Declare as OpenACC routine accordingly

  allocate(buffer(n))

  ! TODO: Copy necessary data to the GPU here
  do istep = 1,nsteps
     ! TODO: Offload this loop to the GPU
     do i = 2,n-1
        buffer(i) = blur(i, in, n)
     enddo

     ! TODO: Offload this loop to the GPU
     do i = 3,n-2
        out(i) = blur(i, buffer, n)
     enddo

     ! TODO: Offload this loop to the GPU
     do i = 1,n
        in(i) = out(i)
     enddo
  enddo

  deallocate(buffer)
end subroutine blur_twice_gpu_nocopies

program main
  use util
  implicit none

  integer pow, n, nsteps, err, i
  real(kind(0d0)), dimension(:), allocatable :: x0, x0_orig, x1, x1_orig
  real(kind(0d0)) :: time_gpu, time_host
  logical:: validate
  pow    = read_arg(1, 20)
  nsteps = read_arg(2, 100)
  n      = 2**pow + 4

  write(*, '(a i0 a f0.6 a)') 'dispersion 1D test of length n = ', n, ' : ', 8.*n/1024**2, 'MB'

  allocate(x0(n), x1(n), x0_orig(n), x1_orig(n), stat=err)
  if (err /= 0) then
     stop 'failed to allocate arrays'
  endif

  x0(1)   = 1.0
  x0(2)   = 1.0
  x0(n-1) = 1.0
  x0(n)   = 1.0
  x0_orig(1)   = 1.0
  x0_orig(2)   = 1.0
  x0_orig(n-1) = 1.0
  x0_orig(n)   = 1.0

  x1(1)   = x0(1)
  x1(2)   = x0(2)
  x1(n-1) = x0(n-1)
  x1(n)   = x0(n)
  x1_orig(1)   = x0(1)
  x1_orig(2)   = x0(2)
  x1_orig(n-1) = x0(n-1)
  x1_orig(n)   = x0(n)

  time_host = get_time()
  call blur_twice_host(nsteps, n, x0_orig, x1_orig)
  time_host = get_time() - time_host

  time_gpu = get_time()
  call blur_twice_gpu_nocopies(nsteps, n, x0, x1)
  time_gpu = get_time() - time_gpu

  ! Validate kernel
  validate = .true.
  do i = 1, n
     if (abs(x1_orig(i) - x1(i)) > 1.e-6) then
        write(*, *) 'item ', i, ' differs (expected, found): ', &
             x1_orig(i), ' != ', x1(i)
     endif
  enddo

  if (validate) then
     write(*, '(a)') '==== success ===='
  else
     write(*, '(a)') '==== failure ===='
  endif

  write(*, '(a f0.6 a f0.6 a)') 'Host version took ', time_host, ' s (', &
       time_host/nsteps, ' s/steps)'
  write(*, '(a f0.6 a f0.6 a)') 'GPU version took ', time_gpu, ' s (',   &
       time_gpu/nsteps, ' s/steps)'

end program main
