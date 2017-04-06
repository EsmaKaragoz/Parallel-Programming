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
    int cur_local_size = end - begin, total;
    MPI_Allreduce(&cur_local_size, &total, 1, MPI_INT, MPI_SUM, comm);

    /*********************************************************************
     *               Broadcast the pivot to each processor               *
     *********************************************************************/

    // Compute the global index of this processor
    int q = total / p; // get the quotient
    int r = total % p; // get the remainder
    int lo;
    if (rank < r + 1) lo = (q + 1) * rank;
    else lo = (q + 1) * r + q * (rank - r);

    // Set the random seed and uniform integer distribution
    srand(0);
    int k = rand() % total;

    // Broadcast the pivot
    int root;
    if (r == 0) root = k / q;
    else if (k < r * (q + 1)) root = k / (q + 1);
    else root = r + (k - r * (q + 1)) / q;

    int pivot = 0;
    if (k >= lo && k < lo + cur_local_size)
    	pivot = begin[k - lo];
    MPI_Bcast(&pivot, 1, MPI_INT, root, comm);

    /*********************************************************************
     *             Partition array locally on each processor             *
     *********************************************************************/
    int boundary = partition(begin, cur_local_size, pivot);
    int small = boundary + 1;

    /*********************************************************************
     *       Gather the info of two subarrays among all processors       *
     *********************************************************************/
    std::vector<int> small_arr(p), cur_size(p);
    MPI_Allgather(&small, 1, MPI_INT, &small_arr[0], 1, MPI_INT, comm);
    MPI_Allgather(&cur_local_size, 1, MPI_INT, &cur_size[0], 1, MPI_INT, comm);

    /*********************************************************************
     *          Transfer the data using All-to-all communication         *
     *********************************************************************/
    int small_sum = std::accumulate(small_arr.begin(), small_arr.end(), 0);

    /* Get the cut point of p processors, small number will be sent to 0, 1, ..., 
     * cutpoint - 1 processors, large number will be sent to the rest processors */
    int cutpoint = ceil(1.0 * p * small_sum / total);
    if (cutpoint == 0) cutpoint++;
    if (cutpoint == p) cutpoint--;

    // Compute the new local size vector
    std::vector<int> new_size(p, 0);
    compute_size(&new_size[0], cutpoint, small_sum, total - small_sum, comm);
    // Compute the send and receive counts
    std::vector<int> reference(new_size), sendcnts(p, 0), recvcnts(p, 0);
    compute_counts(&sendcnts[0], &recvcnts[0], cutpoint, &small_arr[0], &cur_size[0], &reference[0], comm);
    // Compute the send and receive displacement
    std::vector<int> sdispls(p, 0), rdispls(p, 0);
    compute_displs(&sdispls[0], &rdispls[0], &sendcnts[0], &recvcnts[0], comm);

    // All-to-to communication
    int new_local_size = new_size[rank];
    int *rbuf = new int[new_local_size];
    MPI_Alltoallv(begin, &sendcnts[0], &sdispls[0], MPI_INTEGER, rbuf, &recvcnts[0], &rdispls[0], MPI_INTEGER, comm);

    /*********************************************************************
     *            Create new communicator and recursively sort           *
     *********************************************************************/
    MPI_Comm new_comm;
    MPI_Comm_split(comm, (rank < cutpoint), rank, &new_comm);
    parallel_sort(rbuf, rbuf + new_local_size, new_comm);
    MPI_Comm_free(&new_comm);

    /*********************************************************************
     *          Transfer the data using All-to-all communication         *
     *********************************************************************/
    std::vector<int> tmp_sendcnts(p, 0), tmp_recvcnts(p, 0), tmp_sdispls(p, 0), tmp_rdispls(p, 0);
    std::vector<int> tmp_reference(cur_size);
    for (int i = 0, j = 0; i < p; i++) {
        int send = new_size[i];
        while (send > 0) {
            int s = (send <= tmp_reference[j])? send : tmp_reference[j];
            if (i == rank) tmp_sendcnts[j] = s;
            if (j == rank) tmp_recvcnts[i] = s;
            send -= s;
            tmp_reference[j] -= s;
            if (tmp_reference[j] == 0) j++;
        }
    }
    compute_displs(&tmp_sdispls[0], &tmp_rdispls[0], &tmp_sendcnts[0], &tmp_recvcnts[0], comm);

    // All-to-to communication
    MPI_Alltoallv(rbuf, &tmp_sendcnts[0], &tmp_sdispls[0], MPI_INTEGER, begin, &tmp_recvcnts[0], &tmp_rdispls[0], MPI_INTEGER, comm);

    delete [] rbuf;
}


/*********************************************************************
 *             Implement your own helper functions here:             *
 *********************************************************************/

/* Partition an array to two subarrays according to the pivot, and  
 * return the index of the largest number in the first subarrays */
int partition(int* begin, int size, int pivot) {
    if (size == 0) return -1;
    int i = -1, j = size;
    while (true) {
        do {
            i++;
        } while (begin[i] <= pivot);

        do {
            j--;
        } while (begin[j] > pivot);

        if (i >= j) return j;

        std::swap(begin[i], begin[j]);
    }
}

int compute_size(int* new_size, int cutpoint, int small_sum, int large_sum, MPI_Comm comm) {
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);

    // Compute the new quotient and remainder for two subset of processors, and the new local size for each processor
    int squotient = small_sum / cutpoint, sremainder = small_sum % cutpoint;
    int lquotient = large_sum / (p - cutpoint), lremainder = large_sum % (p - cutpoint) + cutpoint;
    for (int i = 0; i < p; i++) {
        if (i < cutpoint) new_size[i] = (i < sremainder)? squotient + 1 : squotient;
        else new_size[i] = (i < lremainder)? lquotient + 1: lquotient;
    }
    return 0;
}

int compute_counts(int* sendcnts, int* recvcnts, int cutpoint, int* small_size, int* local_size, int* new_size, MPI_Comm comm) {
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);

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
        int large_send = local_size[i] - small_size[i];
        while (large_send > 0) {
            int l = (large_send <= new_size[k])? large_send : new_size[k];
            if (i == rank) sendcnts[k] = l;
            if (k == rank) recvcnts[i] = l;
            large_send -= l;
            new_size[k] -= l;
            if (new_size[k] == 0) k++;
        }
    }
    return 0;
}

int compute_displs(int* sdispls, int* rdispls, int* sendcnts, int* recvcnts, MPI_Comm comm) {
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);

    for (int i = 1; i < p; i++) {
        sdispls[i] = sdispls[i - 1] + sendcnts[i - 1];
        rdispls[i] = rdispls[i - 1] + recvcnts[i - 1];
    }
    return 0;
}