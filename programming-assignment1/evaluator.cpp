/*
 * CX 4220 / CSE 6220 Introduction to High Performance Computing
 *              Programming Assignment 1
 * 
 *  Serial polynomial evaluation algorithm function implementations goes here
 * 
 */

double poly_evaluator(const double x, const int n, const double* constants){
    //Implementation
    double ans = 0, sum = 1;
    for (int i = 0; i < n; i++) {
        ans += sum * constants[i];
        sum = sum * x;
    }
    return ans;
}
