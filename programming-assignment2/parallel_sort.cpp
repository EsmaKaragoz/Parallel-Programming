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
    	pivot = begin[k - lo];
    MPI_Bcast(&pivot, 1, MPI_INT, root, comm);
    // MPI_Allreduce(&pivot, &pivot, 1, MPI_INT, MPI_SUM, comm);

    /*********************************************************************
     *             Partition array locally on each processor             *
     *********************************************************************/
    // index of small less than or equal to boundary
    int boundary = partition(begin, local_size, pivot); 
    // Get the numbers of small and large subarray
    int small = boundary + 1, large = local_size - small;

    /*********************************************************************
     *       Gather the info of two subarrays among all processors       *
     *********************************************************************/
    std::vector<int> small_size(p), large_size(p);
    MPI_Allgather(&small, 1, MPI_INT, &small_size, 1, MPI_INT, comm);
    MPI_Allgather(&large, 1, MPI_INT, &large_size, 1, MPI_INT, comm);

    /*********************************************************************
     *          Transfer the data using All-to-all communication         *
     *********************************************************************/
    int small_sum = std::accumulate(small_size.begin(), small_size.end(), 0);
    int large_sum = std::accumulate(large_size.begin(), large_size.end(), 0);

    /* Get the cut point of p processors, small number will be sent to 0, 1, ..., 
     * cutpoint - 1 processors, large number will be sent to the rest processors */
    int cutpoint = p * small_sum / (small_sum + large_sum);
    if (cutpoint == 0) cutpoint++;
    if (cutpoint == p) cutpoint--;

    transfer(begin, cutpoint, small_size, large_size, small_sum, large_sum, comm);

    /*********************************************************************
     *            Create new communicator and recursively sort           *
     *********************************************************************/
    MPI_Comm new_comm;
    MPI_Comm_split(comm, (rank < cutpoint), rank, &new_comm);
    parallel_sort(begin, end, new_comm);
    MPI_Comm_free(&new_comm);

    return;
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
        } while (begin[i] < pivot);

        do {
            j--;
        } while (begin[j] > pivot);

        if (i >= j) return j;

        std::swap(begin[i], begin[j]);
    }
}

/* Transfer data to the correspoding processors */
int transfer(int* sbuf, int cutpoint, int* small_size, int* large_size, int small_sum, int large_sum, MPI_Comm comm) {
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);

    // Compute the new quotient and remainder for two subset of processors, and the new local size for each processor
    int squotient = small_sum / cutpoint, sremainder = small_sum % cutpoint;
    int lquotient = large_sum / (p - cutpoint), lremainder = large_sum % (p - cutpoint) + cutpoint;
    int *new_size = new int[p];
    for (int idx = 0; idx < p; idx++) {
        if (idx < cutpoint) new_size[idx] = (idx < sremainder)? squotient + 1 : squotient;
        else new_size[idx] = (idx < lremainder)? lquotient + 1: lquotient;
    }

    // std::vector<int> change_size(p);
    // for (int i = 0; i < pl i++) {
    //     change_size[i] = (i < cutpoint)? small_size[i] - new_size[i] : large_size[i] - new_size[i];
    // }

    // Compute the prefix sum of small_size and large_size
    // std::vector<int> samll_prefix_sum(p), large_prefix_sum(p);
    // std::partial_sum(small_size.begin(), small_size.end(), samll_prefix_sum);
    // std::partial_sum(large_size.begin(), large_size.end(), large_prefix_sum);

    // std::vector<int> sendcnts(p), recvcnts(p), sdispls(p), rdispls(p);
    int *sendcnts = new int[p], *recvcnts = new int[p];
    int j = 0, k = cutpoint; // denote the current recieving processors
    for (int i = 0; i < p; i++) {
        // Fill the small part
        int small_send = small_size[i];
        while (small_send > 0) {
            int s = (small_send <= new_size[j])? small_send : new_size[j];
            if (i == rank) sendcnts[j] = s;
            if (j == rank) recvcnts[i] = s;
            small_send -= s;
            new_size[j] -= s;
            if (new_size[j] == 0) j++;
        }

        // Fill the large part
        int large_send = large_size[i];
        while (large_send > 0) {
            int l = (large_send <= new_size[k])? large_send : new_size[k];
            if (i == rank) sendcnts[k] = l;
            if (k == rank) recvcnts[i] = l;
            large_send -= l;
            new_size[k] -= l;
            if (new_size[k] == 0) k++;
        }
    }

    int *sdispls = new int[p], *rdispls = new int[p];
    sdispls[0] = 0; 
    rdispls[0] = 0;
    for (int idx = 1; idx < p; idx++) {
        sdispls[idx] = sdispls[idx - 1] + sendcnts[idx - 1];
        rdispls[idx] = rdispls[idx - 1] + recvcnts[idx - 1];
    }

    int* rbuf = new int[new_size[rank]];
    MPI_Alltoallv(sbuf, sendcnts, sdispls, MPI_INTEGER, rbuf, recvcnts, rdispls, MPI_INTEGER, comm);

    delete [] sbuf;
    sbuf = new int[new_size[rank]];
    for (int idx = 0; idx < new_size[rank]; idx++) {
        sbuf[i] = rbuf[i];
    }
    delete [] rbuf;

    delete [] new_size;
    delete [] sendcnts;
    delete [] recvcnts;
    delete [] sdispls;
    delete [] rdispls;
}