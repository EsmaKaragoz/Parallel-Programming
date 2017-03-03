#!/bin/sh

#PBS -q class
#PBS -l node=4:sixcore
#PBS -l walltime=00:10:00
#PBS -N ywu613

# change to our project directory
cd $HOME/programming-assignment1
# hardcode MPI path
MPIRUN=/usr/lib64/openmpi/bin/mpirun
# loop over number of processors (our 4 nodes job can run up to 48)
for p in 2 4 6 8 10 12 14 16 20 24 28 32 36 40 44 48
do
    $MPIRUN -np $p --hostfile $PBS_NODEFILE ./poly-eval -n 5000000 -m 1 -s 6220
done