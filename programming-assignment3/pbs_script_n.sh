#!/bin/sh

#PBS -q class
#PBS -l nodes=jinx2+jinx4+jinx15
#PBS -l walltime=00:10:00
#PBS -N cse6220-prog3

# change to our project directory
cd $HOME/programming-assignment3
# hardcode MPI path
MPIRUN=/usr/lib64/openmpi/bin/mpirun
# loop over number of processors (our 4 nodes job can run up to 48)
for n in 100 500 1000 1500 2000 3000 5000 10000
do
    $MPIRUN -np 16 --hostfile $PBS_NODEFILE ./jacobi -n $n
done