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

using namespace std;

void scatter(const int n, double* scatter_values, int &n_local, double* &local_values, int source_rank, const MPI_Comm comm){
    //Implementation
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);

    int num = broadcast(n, 0, comm);
    int r = num % p;

    if (rank == source_rank) {
        n_local = r==0 ? n/p : n/p+1;
        local_values = (double*) malloc( n_local * sizeof(double) );
        for (int k = 0; k < n_local; k++) local_values[k] = scatter_values[k];
        for (int i = 1, start = n_local; i < p; i++) {
            if (i < r || r == 0) {
                MPI_Send(scatter_values + start, n_local, MPI_DOUBLE, i, i, comm);
                start += n_local;
            } else {
                MPI_Send(scatter_values + start, n_local - 1, MPI_DOUBLE, i, i, comm);
                start += n_local - 1; 
            }
        }
    } else {
        n_local = rank<r ? num/p+1 : num/p;
        local_values = (double*) malloc( n_local * sizeof(double) );
        MPI_Recv(local_values, n_local, MPI_DOUBLE, 0, rank, comm, MPI_STATUS_IGNORE);
    }
}

double broadcast(double value, int source_rank, const MPI_Comm comm){
    //Implementation
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);

    double d = log2(p);
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
    return value;
}

void parallel_prefix(const int n, const double* values, double* prefix_results, const int OP, const MPI_Comm comm){
    //Implementation
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);

    double local = 0.0;
    if (OP == PREFIX_OP_SUM) {
        local = 0.0;
    } else if (OP == PREFIX_OP_PRODUCT) {
        local = 1.0;
    }

    double total = *(prefix_results+n-1);
    double d = log2(p);
    for (int i = 0; i < d; i++) {
        int rank_a = rank ^ int(pow(2, i));
        if (rank_a < p) {
            double recv;
            MPI_Send(&total, 1, MPI_DOUBLE, rank_a, 0, comm);
            MPI_Recv(&recv, 1, MPI_DOUBLE, rank_a, 0, comm, MPI_STATUS_IGNORE);

            if (OP == PREFIX_OP_SUM) {
                total += recv;
            } else if (OP == PREFIX_OP_PRODUCT) {
                total *= recv;
            }

            if (rank_a < rank) {
                if (OP == PREFIX_OP_SUM) {
                    local += recv;
                } else if (OP == PREFIX_OP_PRODUCT) {
                    local *= recv;
                }
            }
        }
    }

    if (OP == PREFIX_OP_SUM) {
        for (int k = 0; k < n; k++) {
            prefix_results[k] += local;
        }
    } else if (OP == PREFIX_OP_PRODUCT) {
        for (int k = 0; k < n; k++) {
            prefix_results[k] *= local;
        }
    }
}

double mpi_poly_evaluator(const double x, const int n, const double* constants, const MPI_Comm comm){
    //Implementation
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);

    // Compute local prefix product, then run parallel prefix
    double* values = (double*) malloc(n*sizeof(double));
    double* prefix = (double*) malloc(n*sizeof(double));
    for (int i = 0; i < n; i++) {
        if (rank == 0 && i == 0) values[i] = 1;
        else values[i] = x;
        prefix[i] = i==0? values[i] : prefix[i-1]*values[i];
    }
    parallel_prefix(n, values, prefix, PREFIX_OP_PRODUCT, comm);

    // Compute local prefix sum, then run parallel prefix
    for (int j = 0; j < n; j++) {
        values[j] = constants[j] * prefix[j];
        prefix[j] = j==0? values[j] : prefix[j-1]+values[j];
    }
    parallel_prefix(n, values, prefix, PREFIX_OP_SUM, comm);

    double result = prefix[n-1];
    if (rank == p - 1) MPI_Send(&result, 1, MPI_DOUBLE, 0, 0, comm);
    if (rank == 0) MPI_Recv(&result, 1, MPI_DOUBLE, p-1, 0, comm, MPI_STATUS_IGNORE);
    return result;
}
