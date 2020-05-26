#!/bin/bash
# not actually a makefile... i don't know how to make these

module purge
module load slurm
module load gcc/6.4.0
module load openmpi/gcc-6/1.10.3
module load netcdf/gcc/openmpi/4.6.1 

rm ncprogram.pyf 
f2py $1 -m ncprogram -h ncprogram.pyf
f2py -c --fcompiler=gfortran -I$NETCDF/include -L$NETCDF/lib -lnetcdff ncprogram.pyf $1
#f2py -c --fcompiler=ifort -I$NETCDF/include -L$NETCDF/lib -lnetcdf fr.pyf fastread.f90

mv ncprogram.cpython-37m-x86_64-linux-gnu.so ncprogram.so

