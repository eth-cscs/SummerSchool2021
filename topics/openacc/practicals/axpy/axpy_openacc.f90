subroutine axpy(n, alpha, x, y)
  integer, intent(in) :: n
  real(kind(0d0)), intent(in) :: alpha
  real(kind(0d0)), intent(in) :: x(n)
  real(kind(0d0)), intent(inout) :: y(n)

  integer i

  !$omp parallel do
  do i = 1,n
     y(i) = y(i) + alpha*x(i)
  enddo

end subroutine axpy

subroutine axpy_gpu(n, alpha, x, y)
  integer, intent(in) :: n
  real(kind(0d0)), intent(in) :: alpha
  real(kind(0d0)), intent(in) :: x(n)
  real(kind(0d0)), intent(inout) :: y(n)

  integer i

  ! TODO: Offload this loop to the GPU
  do i = 1,n
     y(i) = y(i) + alpha*x(i)
  enddo

end subroutine axpy_gpu

program main
  use util
  implicit none

  integer pow, n, err, i
  real(kind(0d0)), dimension(:), allocatable :: x, x_, y, y_
  real(kind(0d0)) :: axpy_start, copyin_start, copyout_start, time_axpy_omp, &
       time_axpy_gpu, time_copyin, time_copyout

  pow = read_arg(1, 16)
  n = 2**pow
  print *, 'memcopy and daxpy test of size', n
  allocate(x(n), y(n), x_(n), y_(n), stat=err)
  if (err /= 0) then
     stop 'failed to allocate arrays'
  endif

  x(:)  = 1.5d0
  y(:)  = 3.0d0
  x_(:) = 1.5d0
  y_(:) = 3.0d0
  axpy_start = get_time()
  call axpy(n, 2d0, x_, y_)
  time_axpy_omp = get_time() - axpy_start

  copyin_start = get_time()
  ! TODO: Copy data to the GPU
  time_copyin = get_time() - copyin_start

  axpy_start = get_time()
  call axpy_gpu(n, 2d0, x, y)
  time_axpy_gpu = get_time() - axpy_start

  copyout_start = get_time()
  ! TODO: Copy out data from the GPU
  time_copyout = get_time() - copyout_start

  print *, '-------'
  print *, 'timings'
  print *, '-------'
  print *, 'axpy (omp) : ', time_axpy_omp, 's'
  print *, 'axpy (gpu) : ', time_axpy_gpu, 's'
  print *, 'copyin     : ', time_copyin, 's'
  print *, 'copyout    : ', time_copyout, 's'
  print *, 'TOTAL      : ', time_axpy_gpu + time_copyin + time_copyout, 's'

  err=0
  !$omp parallel do reduction(+:err)
  do i = 1,n
     if (abs(6d0 - y(i)) > 1d-15) then
        err = err + 1
     endif
  enddo

  if (err > 0) then
     print *, '============ FAILED with ', err, ' errors'
  else
     print *, '============ PASSED'
  endif

  deallocate(x, y)

end program main
