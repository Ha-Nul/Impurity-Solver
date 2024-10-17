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

    MD_OC(double beta,int grid);
    ~MD_OC();

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
    
    double Limit;
    void SetLimit(double value);
    void Setgrid();

    /*
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
    */

    ////////////////////////////////////////////////////////////////////////////////

    vector<double> alp_arr;
    vector<double> gam_arr;

    ////////////////////////////////////////////////////////////////////////////////

    vector<double> tau_grid;
    vector<double> mode_grid;

    int M;
    int t;
    
    double Delta_t;

    double pi = dlib::pi;
    //vector<double> green();
    void Interact_V(double k_cutoff);

    MatrixXd Eigenvector_Even();
    MatrixXd Eigenvalue_Even();
    MatrixXd Eigenvector_Odd();
    MatrixXd Eigenvalue_Odd();
    
    
    void Hamiltonian_N(MatrixXd even, MatrixXd odd);
    void Hamiltonian_loc(MatrixXd a, MatrixXd b);
    
    void CAL_COUP_INT_with_g_arr(double alp, double cutoff);
    void Tilde_g_calculation_function(double alpha, double k_cutoff);

    void Dataoutput(double gamma, double alpha);


    ///////////////////////////////////////////////////////////////////////////////////////////////
    
    vector<double> coup_Arr;
    vector<double> omega_Arr;
    vector<double> INT_Arr;
    //vector<double> k_mode;

    MatrixXd H_N;

    ///////////////////////////////////////////////////////////////////////////////////////////////

    MatrixXd H_loc;

    vector<double> Chi_Arr;
    vector<vector<MatrixXd> > T;
    vector<vector<MatrixXd> > Chi_st;
    vector<MatrixXd> SELF_E;
    vector<MatrixXd> Prop;
    vector<MatrixXd> GELL;

    void readVfunc();
    void readHN(ifstream& ifstr);
    void readHloc(ifstream& ifstr);

    void NCA_self();
    void OCA_self();
    void OCA_T();
    void SELF_Energy();

    MatrixXd round_propagator_ite(const MatrixXd& loc, const vector<MatrixXd>& sigma, const vector<MatrixXd>& ite, int n, int boolean);
    vector<MatrixXd> Propagator(const vector<MatrixXd>& array, const MatrixXd& loc);

    double chemical_poten(MatrixXd prop);

    vector<MatrixXd> Iteration(const int& iteration);

    //////////////////////////////////////////////////////////////////////////////////////////////

    double temp_minpoint(vector<MatrixXd> &arr);
    vector<double> temp_itemin(vector<MatrixXd> &arr, double minpo, double size);

    //////////////////////////////////////////////////////////////////////////////////////////////

    void NCA_Chi_sp(vector<MatrixXd> &ITER);
    void OCA_store(vector<MatrixXd> &ITER);
    void OCA_Chi_sp(vector<MatrixXd> &ITER);
    vector<double> Chi_sp_Function(vector<MatrixXd> Iter);
};

/////////////////////////////////////////////////////////////////////////////////////

