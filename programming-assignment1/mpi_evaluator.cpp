/*
 * CX 4220 / CSE 6220 Introduction to High Performance Computing
 *              Programming Assignment 1
 * 
 *  MPI polynomial evaluation algorithm function implementations go here
 * 
 */

#include "mpi_evaluator.h"
#include "const.h"

void scatter(const int n, double* scatter_values, int &n_local, double* &local_values, int source_rank, const MPI_Comm comm){
    //Implementation
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);

    int num, r;
    if (rank == source_rank) {
        r = n % p;
        n_local = r==0? n/p : n/p+1;
        for (int i = 0, start = 0; i < p; i++) {
            if (i > 0) MPI_Send(&n, 1, MPI_INT, i, i, comm);
            if (i < r) {
                MPI_Send(scatter_values + start, n_local, MPI_DOUBLE, i, i, comm);
                start += n_local;
            } else {
                MPI_Send(scatter_values + start, n_local - 1, MPI_DOUBLE, i, i, comm);
                start += n_local - 1; 
            }
        }
        local_values = (double*) malloc( n_local * sizeof(double) );
        MPI_Recv(local_values, n_local, MPI_DOUBLE, 0, 0, comm, MPI_STATUS_IGNORE);
    } else {
        MPI_Recv(&num, 1, MPI_INT, 0, rank, comm, MPI_STATUS_IGNORE);
        r = num % p;
        n_local = num / p;
        if (rank < r) n_local = n_local + 1;
        local_values = (double*) malloc( n_local * sizeof(double) );
        MPI_Recv(local_values, n_local, MPI_DOUBLE, 0, rank, comm, MPI_STATUS_IGNORE);
    }
    // if (rank == 0) n_local += 1;
    for (int j = 0; j < n_local; j++) {
        std::cout << local_values[j] << " ";
    }
    std::cout << std::endl;
}

double broadcast(double value, int source_rank, const MPI_Comm comm){
    //Implementation

    return 0;
}

void parallel_prefix(const int n, const double* values, double* prefix_results, const int OP, const MPI_Comm comm){
    //Implementation

}

double mpi_poly_evaluator(const double x, const int n, const double* constants, const MPI_Comm comm){
    //Implementation

    return 0;
}
