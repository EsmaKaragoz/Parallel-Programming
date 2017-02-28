/*
 * CX 4220 / CSE 6220 Introduction to High Performance Computing
 *              Programming Assignment 1
 * 
 *  MPI polynomial evaluation algorithm function implementations go here
 * 
 */

#include "mpi_evaluator.h"
#include "const.h"
#include <math.h>

void scatter(const int n, double* scatter_values, int &n_local, double* &local_values, int source_rank, const MPI_Comm comm){
    //Implementation
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);

    int num = broadcast(n, 0, comm);
    int r = num % p;

    if (rank == source_rank) {
        n_local = r==0 ? n/p : n/p+1;
        for (int i = 0, start = 0; i < p; i++) {
            if (i < r || r == 0) {
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
        n_local = rank<r ? num/p+1 : num/p;
        local_values = (double*) malloc( n_local * sizeof(double) );
        MPI_Recv(local_values, n_local, MPI_DOUBLE, 0, rank, comm, MPI_STATUS_IGNORE);
    }
    for (int j = 0; j < n_local; j++) {
        std::cout << local_values[j] << " ";
    }
    std::cout << std::endl;
}

double broadcast(double value, int source_rank, const MPI_Comm comm){
    //Implementation
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);

    int d = ceil(log2(p));
    bool flag = rank==source_rank ? true : false;
    for (int i = 0; i < d; i++) {
        int rank_a = rank ^ int(pow(2, i));
        if (rank < int(pow(2, i+1)) && rank_a < p) {
            if (flag) MPI_Send(&value, 1, MPI_DOUBLE, rank_a, 0, comm);
            else {
                MPI_Recv(&value, 1, MPI_DOUBLE, rank_a, 0, comm, MPI_STATUS_IGNORE);
                flag = true;
            }
        }
    }

    std::cout << "Rank = " << rank << " and value is " << value << std::endl;

    return value;
}

void parallel_prefix(const int n, const double* values, double* prefix_results, const int OP, const MPI_Comm comm){
    //Implementation

}

double mpi_poly_evaluator(const double x, const int n, const double* constants, const MPI_Comm comm){
    //Implementation

    return 0;
}
