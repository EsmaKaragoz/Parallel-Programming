#!/bin/sh

#PBS -q class
#PBS -l nodes=4:sixcore
#PBS -l walltime=00:10:00
#PBS -N cse6220-prog2

# change to our project directory
cd $HOME/programming-assignment2
# hardcode MPI path
MPIRUN=/usr/lib64/openmpi/bin/mpirun
# loop over number of processors (our 4 nodes job can run up to 48)
for p in 6 12 18 24 30 36 42 48
do
    for n in 10 100 1000 10000 100000 
    do
        $MPIRUN -np $p --hostfile $PBS_NODEFILE ./sort -t -n n -o output.txt
    done
done