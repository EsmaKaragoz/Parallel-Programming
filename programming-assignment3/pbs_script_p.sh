#!/bin/sh

#PBS -q class
#PBS -l nodes=jinx11+jinx12+jinx15
#PBS -l walltime=00:10:00
#PBS -N cse6220-prog3

# change to our project directory
cd $HOME/programming-assignment3
# hardcode MPI path
MPIRUN=/usr/lib64/openmpi/bin/mpirun
# loop over number of processors (our 4 nodes job can run up to 48)
for p in 1 4 9 16 25 36
do
    $MPIRUN -np $p --hostfile $PBS_NODEFILE ./jacobi -n 10000
done