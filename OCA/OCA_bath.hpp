#include<iostream>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <vector>
#include <cmath>
#include <iomanip>
#include <string>
#include <const.h>

using namespace std;
using namespace Eigen;

class MD_OC //MAIN_DEF_OCA
{
private:

    vector<double> linspace(const double& min, const double& max, int n)
    {
        vector<double> result;
        // vector iterator
        int iterator = 0;

        for (int i = 0; i <= n - 2; i++)
        {
            double temp = min + i * (max - min) / (floor((double)n) - 1);
            result.insert(result.begin() + iterator, temp);
            iterator += 1;
        }

        //iterator += 1;

        result.insert(result.begin() + iterator, max);
        return result;
    }

    vector<MatrixXd> convolve(const vector<MatrixXd>& Signal,
        const vector<MatrixXd>& Kernel, int n, int i)
    {
        size_t SignalLen = i;
        size_t KernelLen = Kernel.size();
        size_t ResultLen = SignalLen + KernelLen - 1;

        vector<MatrixXd> Result(ResultLen, MatrixXd::Zero(n, n));

        for (size_t n = 0; n < ResultLen; ++n)
        {
            size_t kmin = (n >= KernelLen - 1) ? n - (KernelLen - 1) : 0;
            size_t kmax = (n < SignalLen - 1) ? n : SignalLen - 1;

            for (size_t k = kmin; k <= kmax; k++)
            {
                Result[n] += Signal[k] * Kernel[n - k];
            }
        }

        return Result;
    }

    MatrixXd Matrix_Odd(int n, double r)
    {
        MatrixXd BASE_MAT=MatrixXd::Zero(n,n);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++)
            {
                try
                {
                    if (i == j) {
                        BASE_MAT(i, j) = pow((i + 1), 2);
                    }
                    if (abs(i - j) == 1) {
                        BASE_MAT(i, j) = -r / 2.0;
                    }
                }
                catch (...) {}
            }
        }
        return BASE_MAT;
    }

    MatrixXd Matrix_Even(int n, double r)
    {
        MatrixXd BASE_MAT=MatrixXd::Zero(n,n);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++)
            {
                try
                {
                    if (i == j) {
                        BASE_MAT(i, j) = pow(i, 2);
                    }
                    if (abs(i - j) == 1) {
                        BASE_MAT(i, j) = -r / 2.0;
                    }
                }
                catch (...) {}
            }
        }
        BASE_MAT(0, 1) = -r / sqrt(2);
        BASE_MAT(1, 0) = -r / sqrt(2);

        return BASE_MAT;
    }

    
public:

    vector<double> tau_grid = linspace(0, 5, 201);
    vector<double> mode_grid = linspace(1,100,100);
    static int M;
    static int t;
    double Delta_t = tau_grid[1] - tau_grid[0];

    double pi = dlib::pi;

    static MatrixXd H_N;

    void CAL_COUP_INT_with_g_arr(double alpha, double k_cutoff);

    vector<double> green(vector<double> tau);
    void Tilde_g_calculation_function(double alpha, double k_cutoff);
    vector<double> Interact_V();

    MatrixXd Eigenvector_Even();
    MatrixXd Eigenvalue_Even();
    MatrixXd Eigenvector_Odd();
    MatrixXd Eigenvalue_Odd();

    MatrixXd Hamiltonian_N(MatrixXd even, MatrixXd odd);
    vector<MatrixXd> Hamiltonian_exp(MatrixXd a, MatrixXd b);
    MatrixXd Hamiltonian_loc(MatrixXd a, MatrixXd b);

    void NCA_self(const MatrixXd& N, const vector<MatrixXd>& H_exp, const vector<double>& V);
    void OCA_self(const vector<MatrixXd>& H_exp);
    void OCA_T(const MatrixXd& N,const vector<MatrixXd>& H_exp, const vector<double>& V);
    void SELF_Energy(vector<MatrixXd> &Prop);

    MatrixXd round_propagator_ite(const MatrixXd& loc, const vector<MatrixXd>& sigma, const vector<MatrixXd>& ite, int n, int boolean);
    vector<MatrixXd> Propagator(const vector<MatrixXd>& array, const MatrixXd& loc);

    double chemical_poten(MatrixXd prop);

    vector<MatrixXd> Iteration(const int& iteration);

    void NCA_Chi_sp(vector<MatrixXd> ITER);
    void OCA_store(vector<MatrixXd> ITER);
    void OCA_Chi_sp(vector<MatrixXd> ITER);
    vector<double> Chi_sp_Function(vector<MatrixXd> Iter);
};

/////////////////////////////////////////////////////////////////////////////////////

