/**
 * @file    parallel_sort.cpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements the parallel, distributed sorting function.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */

#include "parallel_sort.h"

// implementation of your parallel sorting
void parallel_sort(int* begin, int* end, MPI_Comm comm) {
    // Obtain the total number of processors and the rank of this processor
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);

    if (p == 1) {
        std::sort(begin, end);
        return;
    }

    // Get the size of the local array, and the size of the global array
    int local_size = end - begin, total_size;
    MPI_Allreduce(&local_size, &total_size, 1, MPI_INT, MPI_SUM, comm);

    /*********************************************************************
     *               Broadcast the pivot to each processor               *
     *********************************************************************/

    // Compute the global index of this processor
    int q = total_size / p; // get the quotient
    int r = total_size % p; // get the remainder
    int lo;
    if (rank < r + 1) lo = (q + 1) * rank;
    else lo = (q + 1) * r + q * (rank - r);

    // Set the random seed and uniform integer distribution
    std::default_random_engine generator(6220);
    std::uniform_int_distribution<int> distribution(0, total_size);
    int k = distribution(generator);

    // Broadcast the pivot
    int root;
    if (r == 0) root = k % q;
    else if (k < r * (q + 1)) root = k % (q + 1);
    else root = r + (k - r * (q + 1)) % q;

    int pivot = 0;
    if (k >= lo && k < lo + local_size)
    	pivot = &(begin + k - lo);
    MPI_Bcast(&pivot, 1, MPI_DOUBLE, root, comm);
    // MPI_Allreduce(&pivot, &pivot, 1, MPI_INT, MPI_SUM, comm);

    /*********************************************************************
     *             Partition array locally on each processor             *
     *********************************************************************/
    int cutpoint = partition(begin, local_size, pivot);
    int count[2] = {cutpoint + 1, local_size - cutpoint - 1};

    /*********************************************************************
     *        Gather the info of two subarrays among all processors      *
     *********************************************************************/

}


/*********************************************************************
 *             Implement your own helper functions here:             *
 *********************************************************************/

/* Partition an array to two subarrays according to the pivot, and  
 * return the index of the largest number in the first subarrays */
int partition(int* begin, int local_size, int pivot) {
    int i = -1, j = local_size;
    while (true) {
        do {
            i++;
        } while (&(begin+i) < pivot);

        do {
            j--;
        } while (&(begin+j) > pivot);

        if (i >= j) return j;

        std::swap(&(begin+i), &(begin+j));
    }
}
