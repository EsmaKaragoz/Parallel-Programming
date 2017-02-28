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

    double total = *(values+n-1);
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
                    for (int k = 0; k < n; k++) prefix_results[k] += local;
                } else if (OP == PREFIX_OP_PRODUCT) {
                    local *= recv;
                    for (int k = 0; k < n; k++) prefix_results[k] *= local;
                } 
            }
        }
    }
}

double mpi_poly_evaluator(const double x, const int n, const double* constants, const MPI_Comm comm){
    //Implementation
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);

    // First parallel prefix to compute x^i, x^i+1, ..., x^i+n-1
    double* values = (double*) malloc(n*sizeof(double));
    double* prefix = (double*) malloc(n*sizeof(double));
    for (int i = 0; i < n; i++) {
        if (rank == 0) values[i] = i==0? 1 : values[i-1]*x;
        else values[i] = i==0? x : values[i-1]*x;
        prefix[i] = values[i];
    }
    parallel_prefix(n, values, prefix, PREFIX_OP_PRODUCT, comm);

    // Second parallel prefix to compute a_i*x^i + a_i+1*x^i+1 + ... + a_i+n-1*x^i+n-1
    for (int j = 0; j < n; j++) {
        double temp = constants[j]*prefix[j];
        values[j] = j==0? temp : values[j-1]+temp;
        prefix[j] = values[j];
    }

    parallel_prefix(n, values, prefix, PREFIX_OP_SUM, comm);

    if (rank == p - 1) {
        cout << "Parallel prefix finishes!" << endl;
        for (int k = 0; k < n; k++) {
            cout << prefix[k] << " ";
        }
        cout << endl;
        return prefix[n-1];
    }
    return prefix[n-1];
}
