module util
  use omp_lib

  implicit none

contains

  subroutine swap(a, b)
    real(kind(0d0)), dimension(:), intent(inout) :: a, b
    real(kind(0d0)), dimension(size(a)) :: tmp
    tmp = a
    a = b
    b = tmp
  end subroutine swap

  function read_arg(nth_arg,default)
    ! get the command line argument, this is f2003
    integer read_arg
    integer,intent(in) :: nth_arg,default
    character(len=32) :: arg, trimmed_arg
    integer count

    ! but only get the first one, assuming this is convertable to integer
    count = command_argument_count()
    if (count >= nth_arg) then
       call get_command_argument(nth_arg, arg)
       trimmed_arg = trim(arg)
       read(trimmed_arg, '(I10)') read_arg
    else
       read_arg = default
    endif

    return
  end function read_arg

  !--------------------------------------------
  function get_time()
    real(kind(0d0)) :: get_time

    get_time = omp_get_wtime()
    return
  end function get_time

end module util
