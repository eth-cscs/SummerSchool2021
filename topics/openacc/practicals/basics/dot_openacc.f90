subroutine dotprod_gpu(n, x, y, res)
  integer, intent(in) :: n
  real(kind(0d0)), intent(in) :: x(n)
  real(kind(0d0)), intent(in) :: y(n)
  real(kind(0d0)), intent(out) :: res

  res = 0
  ! TODO: Offload this loop to the GPU
  do i = 1,n
     res = res + x(i)*y(i)
  enddo

end subroutine dotprod_gpu

subroutine dotprod_host(n, x, y, res)
  integer, intent(in) :: n
  real(kind(0d0)), intent(in) :: x(n)
  real(kind(0d0)), intent(in) :: y(n)
  real(kind(0d0)), intent(out) :: res

  res = 0
  !$omp parallel do reduction(+:res)
  do i = 1,n
     res = res + x(i)*y(i)
  enddo

end subroutine dotprod_host

program main
  use util
  implicit none

  integer n, pow, err
  real(kind(0d0)), dimension(:), allocatable :: x, y
  real(kind(0d0)) :: result, expected
  real(kind(0d0)) :: time_gpu, time_host

  pow = read_arg(1, 2)
  n = 2**pow

  write(*, '(a i0)') 'dot product of length n = ', n
  allocate(x(n), y(n), stat=err)
  if (err /= 0) then
     stop 'failed to allocate arrays'
  endif

  call random_seed()

  x(:) = 2.0d0
  call random_number(y)
  y(:) = 10*y(:)

  time_host = get_time()
  call dotprod_host(n, x, y, expected)
  time_host = get_time() - time_host

  time_gpu = get_time()
  call dotprod_gpu(n, x, y, result)
  time_gpu = get_time() - time_gpu

  if (abs(expected - result) > 1.e-6) then
     write(*, '(a e0.6 a e0.6 a)') 'expected ', expected, ' got, ', result, &
          ': failure'
  else
     write(*, '(a e0.6 a e0.6 a)') 'expected ', expected, ' got, ', result, &
          ': success'
  endif

end program main
