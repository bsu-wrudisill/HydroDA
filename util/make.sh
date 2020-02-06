#!/bin/bash
# not actually a makefile... i don't know how to make these

module purge
module load slurm
module load gcc/6.4.0
module load openmpi/gcc-6/1.10.3
module load netcdf/gcc/openmpi/4.6.1 

rm fr.pyf
f2py fastread.f90 -m fr -h fr.pyf
f2py -c --fcompiler=gfortran -I$NETCDF/include -L$NETCDF/lib -lnetcdff fr.pyf fastread.f90
#f2py -c --fcompiler=ifort -I$NETCDF/include -L$NETCDF/lib -lnetcdf fr.pyf fastread.f90
