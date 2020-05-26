module test
contains
subroutine readnc(file_name, x, y, outputfile)
  use netcdf
  implicit none 
  character (len = *) :: file_name
  character (len = *) :: outputfile
  character (len = *), parameter :: varname = "RAINNC"
  integer :: featureid, x, y, lat, lon, ncid, varid, status
  real :: data(x,y), output(x,y)
  
  !f2py intent(out) output
  
  ! open the file 
  status = nf90_open(file_name, NF90_NOWRITE, ncid)
  
  !! get the variable 
  status = nf90_inq_varid(ncid, varname, varid)
  
  !! get the data 
  status = nf90_get_var(ncid, varid, data)
  
  !! now we can access the data ... i think
  output = data(x,y)
 
  do lat = 1, x
        do lon =1, y
                print *, data(lat,lon)
        end do
  end do 
  
  !  ! write the data to a text file 
  !open (10, file=outputfile, status='unknown', position='append')
  !write(10, *) FILE_NAME, ',', output
  !close(10)
  
  ! close the netcdf file  
  status = nf90_close(ncid)
  
end subroutine readnc
end module test 
