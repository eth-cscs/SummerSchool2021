module diffusion2d
  implicit none

contains
  subroutine diffusion_gpu(x0, x1, nx, ny, dt)
    integer, intent(in) :: nx, ny
    real(kind(0d0)), intent(in) :: dt
    real(kind(0d0)), dimension(:), intent(in)  :: x0
    real(kind(0d0)), dimension(:), intent(out) :: x1
    integer :: i, j, pos, width

    width = nx + 2

    !$acc parallel loop present(x0, x1) collapse(2) private(i,j)
    do j = 2, ny+1
       do i = 2, nx+1
          pos = i + (j-1)*width;
          x1(pos) = x0(pos) + dt*(-4.0*x0(pos) + &
               x0(pos-width) + x0(pos+width) + x0(pos-1) + x0(pos+1))
       enddo
    enddo
  end subroutine diffusion_gpu

  subroutine copy_gpu(dst, src, n)
    integer, intent(in) :: n
    real(kind(0d0)), intent(out) :: dst(n)
    real(kind(0d0)), intent(in)  :: src(n)
    integer :: i

    !$acc parallel loop present(dst, src)
    do i = 1, n
       dst(i) = src(i)
    enddo
  end subroutine copy_gpu


  subroutine fill_gpu(arr, val, n)
    integer, intent(in) :: n
    real(kind(0d0)), intent(in)  :: val
    real(kind(0d0)), intent(out) :: arr(n)
    integer :: i

    !$acc parallel loop present(arr)
    do i = 1, n
       arr(i) = val
    enddo
  end subroutine fill_gpu

  subroutine write_to_file(nx, ny, data)
    integer, intent(in) :: nx, ny
    real(kind(0d0)), intent(in) :: data(nx*ny)

    integer :: i

    open(10, file='output.bin', status='replace', access='stream', &
         form='unformatted')
    write(10) data
    close(10)

    open(20, file='output.bov')
    write(20, '(a)') 'TIME: 0.0'
    write(20, '(a)') 'DATA_FILE: output.bin'
    write(20, '(a i0 a i0 a)') 'DATA_SIZE: ', nx, ' ', ny, ' 1'
    write(20, '(a)') 'DATA_FORMAT: DOUBLE'
    write(20, '(a)') 'VARIABLE: phi'
    write(20, '(a)') 'DATA_ENDIAN: LITTLE'
    write(20, '(a)') 'CENTERING: nodal'
    write(20, '(a)') 'BRICK_SIZE: 1.0 1.0 1.0'

  end subroutine write_to_file

end module diffusion2d
