#!/bin/sh

#PBS -q class
#PBS -l nodes=jinx8+jinx11+jinx12
#PBS -l walltime=00:10:00
#PBS -N cse6220-prog3

# change to our project directory
cd $HOME/programming-assignment3
# hardcode MPI path
MPIRUN=/usr/lib64/openmpi/bin/mpirun
# loop over number of processors (our 4 nodes job can run up to 48)
for d in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
    $MPIRUN -np 16 --hostfile $PBS_NODEFILE ./jacobi -n 10000 -d $d
done