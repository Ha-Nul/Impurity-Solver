#include<iostream>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <vector>
#include <cmath>

using namespace std;
using namespace Eigen;

class Testing
{
    private:

        vector<double> linspace(const double &min,const double &max, int n)
        {
            vector<double> result;
            // vector iterator
            int iterator = 0;

            for (int i = 0; i <= n-2; i++)	
            {
                double temp = min + i*(max-min)/(floor((double)n) - 1);
                result.insert(result.begin() + iterator, temp);
                iterator += 1;
            }

            //iterator += 1;

            result.insert(result.begin() + iterator, max);
            return result;
        }

        MatrixXd Matrix_Odd(int n, double r)
        {
            MatrixXd Matrix1(n,n);

            for (int i=0;i < n;i++){
                for (int j=0;j < n;j++)
                {
                    try
                    {
                        if(i==j){
                            Matrix1(i,j) = pow((i+1),2);
                        }
                        if(abs(i - j) == 1){
                            Matrix1(i,j) = -r/2.0;
                        }
                    }
                    catch (...) {}
                }
            }
            return Matrix1;
        }

        MatrixXd Matrix_Even(int n, double r)
        {
            MatrixXd Matrix1(n,n);

            for (int i=0;i < n;i++){
                for (int j=0;j < n;j++)
                {
                    try
                    {
                        if(i==j){
                            Matrix1(i,j)= pow(i,2);
                        }
                        if(abs(i - j) == 1){
                            Matrix1(i,j) = -r/2.0;
                        }
                    }
                    catch (...) {}
                }
            }
            Matrix1(0,1) = -r/sqrt(2);
            Matrix1(1,0) = -r/sqrt(2);

            return Matrix1;
        }

        vector<double> tau_grid = linspace(0,1,100);
        //vector<int> k_grid = {1,1,1,1,1,1,1,1,1,1};
        int k = tau_grid.size();

    public:

        vector<double> grid = linspace(0,1,100);
        vector<double> green(double tau);
        vector<double> coupling(double v, double g, double W);
        vector<double> Interact(vector<double> coupling, vector<double> tau);
        vector<double> Interact_V(vector<double> couplint, vector<double> tau, double omega);

        MatrixXd Eigenvector_Even();
        MatrixXd Eigenvalue_Even();
        MatrixXd Eigenvector_Odd();
        MatrixXd Eigenvalue_Odd();

        MatrixXd Hamiltonian_N(MatrixXd even, MatrixXd odd, double g);
        vector<MatrixXd> Hamiltonian_exp(MatrixXd a, MatrixXd b);
        MatrixXd Hamiltonian_loc(MatrixXd a, MatrixXd b);
        MatrixXd Hamiltonian_loc_ite(MatrixXd a, MatrixXd b,const double &lambda);

        MatrixXd round_propagater_ite(const MatrixXd &loc, const vector<MatrixXd> &sigma, const MatrixXd &ite, int weight);
        vector<MatrixXd> Sigma(const MatrixXd &N,const vector<MatrixXd> &H_exp, const vector<double> &V);
        vector<MatrixXd> Propagator(int weight ,const vector<MatrixXd> &array , const MatrixXd &loc);

        double chemical_poten(MatrixXd prop);

        vector<MatrixXd> Iteration(const int &weight, int iteration);
        vector<double> TestingIteration(const int &n, int testingint);

        vector<double> Chi_sp(const int &weight, int iteration);

};