/**
 * @file    mpi_jacobi.cpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements MPI functions for distributing vectors and matrixes,
 *          parallel distributed matrix-vector multiplication and Jacobi's
 *          method.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */

#include "mpi_jacobi.h"
#include "jacobi.h"
#include "utils.h"

#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <vector>
#include <iostream>

/*
 * TODO: Implement your solutions here
 */


void distribute_vector(const int n, double* input_vector, double** local_vector, MPI_Comm comm) {

    // Get the Cartesian topology information
    int dims[2], periods[2], coords[2];
    MPI_Cart_get(comm, 2, dims, periods, coords);

    // Create column communicators 
    MPI_Comm comm_col;
    int remain_dims[2] = {true, false};
    MPI_Cart_sub(comm, remain_dims, &comm_col);

    if (coords[1] == 0) {


        // std::cout << coords[0] << " " << coords[1] << std::endl;
        // if (coords[0] == 0 && coords[1] == 0) {
        //     for (int i = 0; i < n; i++) {
        //         std::cout << input_vector[i] << " ";
        //     }
        //     std::cout << std::endl;
        // }
        

        // Compute the number of elements to send to each processor
        int* sendcounts = new int[dims[0]];
        int* displs = new int[dims[0]];

        for (int i = 0; i < dims[0]; i++) {
            sendcounts[i] = block_decompose(n, dims[0], i);
        }

        displs[0] = 0;
        for (int i = 1; i < dims[0]; i++) {
            displs[i] = displs[i - 1] + sendcounts[i - 1];
        }

        // Compute local size for processors in the first column
        int local_size = sendcounts[coords[0]];
        (*local_vector) = new double[local_size];

        // Get the rank of root in the first column communicator
        int root_rank;
        int root_coords[2] = {0, 0};
        MPI_Cart_rank(comm_col, root_coords, &root_rank);


        // std::cout << root_rank << std::endl;
        // for (int i = 0; i < dims[0]; i++) {
        //     std::cout << sendcounts[i] << " ";
        // }
        // std::cout << std::endl;
        // for (int i = 0; i < dims[0]; i++) {
        //     std::cout << displs[i] << " ";
        // }
        // std::cout << std::endl;

        // Scatter values to different processors
        MPI_Scatterv(input_vector, sendcounts, displs, MPI_DOUBLE, *local_vector, local_size, MPI_DOUBLE, root_rank, comm_col);


        // for (int i = 0; i < local_size; i++) {
        //     std::cout << (*local_vector)[i] << " ";
        // }
        // std::cout << std::endl;


        delete [] sendcounts;
        delete [] displs;
    }

    // Free the column communicator
    MPI_Comm_free(&comm_col);

    return;
}


// gather the local vector distributed among (i,0) to the processor (0,0)
void gather_vector(const int n, double* local_vector, double* output_vector, MPI_Comm comm) {

    // Get the Cartesian topology information
    int dims[2], periods[2], coords[2];
    MPI_Cart_get(comm, 2, dims, periods, coords);

    // Create column communicators 
    MPI_Comm comm_col;
    int remain_dims[2] = {true, false};
    MPI_Cart_sub(comm, remain_dims, &comm_col);

    if (coords[1] == 0) {
        // Compute the number of elements to send to each processor
        int* recvcounts = new int[dims[0]];
        int* displs = new int[dims[0]];

        for (int i = 0; i < dims[0]; i++) {
            recvcounts[i] = block_decompose(n, dims[0], i);
        }

        displs[0] = 0;
        for (int i = 1; i < dims[0]; i++) {
            displs[i] = displs[i - 1] + recvcounts[i - 1];
        }

        // Compute local size for processors in the first column
        int local_size = recvcounts[coords[0]];

        // Get the rank of root in the first column communicator
        int root_rank;
        int root_coords[2] = {0, 0};
        MPI_Cart_rank(comm_col, root_coords, &root_rank);

        // Gather values from different processors
        MPI_Gatherv(local_vector, local_size, MPI_DOUBLE, output_vector, recvcounts, displs, MPI_DOUBLE, root_rank, comm_col);

        delete [] recvcounts;
        delete [] displs;
    }

    // Free the column communicator
    MPI_Comm_free(&comm_col);

    return;
}

void distribute_matrix(const int n, double* input_matrix, double** local_matrix, MPI_Comm comm) {
    // TODO
}


void transpose_bcast_vector(const int n, double* col_vector, double* row_vector, MPI_Comm comm)
{
    // TODO
}


void distributed_matrix_vector_mult(const int n, double* local_A, double* local_x, double* local_y, MPI_Comm comm)
{
    // TODO
}

// Solves Ax = b using the iterative jacobi method
void distributed_jacobi(const int n, double* local_A, double* local_b, double* local_x,
                MPI_Comm comm, int max_iter, double l2_termination)
{
    // TODO
}


// wraps the distributed matrix vector multiplication
void mpi_matrix_vector_mult(const int n, double* A,
                            double* x, double* y, MPI_Comm comm)
{
    // distribute the array onto local processors!
    double* local_A = NULL;
    double* local_x = NULL;
    distribute_matrix(n, &A[0], &local_A, comm);
    distribute_vector(n, &x[0], &local_x, comm);

    // allocate local result space
    double* local_y = new double[block_decompose_by_dim(n, comm, 0)];
    distributed_matrix_vector_mult(n, local_A, local_x, local_y, comm);

    // gather results back to rank 0
    gather_vector(n, local_y, y, comm);
}

// wraps the distributed jacobi function
void mpi_jacobi(const int n, double* A, double* b, double* x, MPI_Comm comm,
                int max_iter, double l2_termination)
{
    // distribute the array onto local processors!
    double* local_A = NULL;
    double* local_b = NULL;
    distribute_matrix(n, &A[0], &local_A, comm);
    distribute_vector(n, &b[0], &local_b, comm);

    // allocate local result space
    double* local_x = new double[block_decompose_by_dim(n, comm, 0)];
    distributed_jacobi(n, local_A, local_b, local_x, comm, max_iter, l2_termination);

    // gather results back to rank 0
    gather_vector(n, local_x, x, comm);
}
