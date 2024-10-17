#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <vector>
#include <cmath>
#include <OCA_bath.hpp>
#include <string>
#include <sstream>

using namespace std;
using namespace Eigen;

///////////////////////////////////////////////////////////////////////

double g_ma;
double alpha;
int sys;

///////////////////////////////////////////////////////////////////////

MD_OC::MD_OC(double beta, int grid)
     : tau_grid(linspace(0,beta,grid)) , t(grid-1)
{
    mode_grid = linspace(1,30000,30000);

    Delta_t = tau_grid[1] - tau_grid[0];

    M = mode_grid.size();
    t = tau_grid.size();
    H_N = MatrixXd::Zero(3,3);
    H_loc = MatrixXd::Zero(3,3);
    
    coup_Arr.resize(M);
    omega_Arr.resize(M);
    INT_Arr.resize(t);
    //vector<double> k_mode(100,1);

}

MD_OC::~MD_OC()
{
    //blank
}

/*
vector<double> MD_OC::green()
{
    vector<double> bose_dist(t);

    for (int i = 0; i < t; i++)
    {
        bose_dist[i] = 1 / (exp(tau_grid[t - 1] * k_mode[i]) - 1);
    }

    vector<double> green(t);

    for (int j = 0; j < t; j++)
    {
        green[j] = ((bose_dist[j] + 1) * exp(-1 * k_mode[j] * tau_grid[j]) + (bose_dist[j]) * exp(k_mode[j] * tau_grid[j]));
    }

    return green;
}
*/

void MD_OC::Tilde_g_calculation_function(double alpha, double k_cutoff)
{
    omega_Arr.resize(M);
    coup_Arr.resize(M);
    double nu = pi * k_cutoff / alpha;
    
    for (int i=0; i < M; i++)
    {
        //omega_Arr[i] = k_cutoff * (mode_grid[i]/mode_grid[M-1]);
        //coup_Arr[i] = sqrt((2 * k_cutoff / (alpha * M)) * (omega_Arr[i] / (1 + pow(nu * omega_Arr[i] / k_cutoff,2))));

        //simpson formulae
        omega_Arr[0] = -0.05;
        omega_Arr[i] = (mode_grid[i]/mode_grid[M-1]); // fix to x to adjust simpson's rule
        coup_Arr[i] = sqrt((2 * k_cutoff / (alpha)) * ( k_cutoff * omega_Arr[i] / (1 + pow(nu * omega_Arr[i],2)))); // fix to adjust simpson's rule
    }

    if (alpha == 0)
    {
        for (int i=0; i < M; i++)
        {
            coup_Arr[i] = 0;
        }
    }
}


////////////////////////////////////////////////////////////////////////////////////

void MD_OC::Interact_V(double k_cutoff)
{
    for (int i = 0; i < t; i++)
    {
        for (int j = 0; j < M ;j++)
        {
            //INT_Arr[i] += -pow(coup_Arr[j],2) * cosh((tau_grid[i] - tau_grid[t - 1] / 2) * omega_Arr[j])/sinh(tau_grid[t - 1] * omega_Arr[j] / 2); //caution for sign
            
            //simpson formulae
            
            if(j == 0 || j == M-1)
            {
                INT_Arr[i] += - ( 1.0 /( 3 * M) ) * pow(coup_Arr[j],2) * cosh((tau_grid[i] - tau_grid[t - 1] / 2) * k_cutoff *  omega_Arr[j])/sinh(tau_grid[t - 1] * k_cutoff * omega_Arr[j] / 2);
            }

            else if (j%2 != 0)
            {
                INT_Arr[i] += - ( 1.0 /(3 * M)) * 4 * pow(coup_Arr[j],2) * cosh((tau_grid[i] - tau_grid[t - 1] / 2) * k_cutoff *  omega_Arr[j])/sinh(tau_grid[t - 1] * k_cutoff * omega_Arr[j] / 2);
            }

            else if (j%2 == 0)
            {
                INT_Arr[i] += - (1.0/(3 * M)) * 2 * pow(coup_Arr[j],2) * cosh((tau_grid[i] - tau_grid[t - 1] / 2) * k_cutoff *  omega_Arr[j])/sinh(tau_grid[t - 1] * k_cutoff * omega_Arr[j] / 2);
            }
            
        }
        //INT_Arr[i] += -0.05;
    }
}

////////////////////////////////////////////////////////////////////////////////////

MatrixXd MD_OC::Eigenvector_Even()
{
    MatrixXd a;

    SelfAdjointEigenSolver<MatrixXd> es(Matrix_Even(sys, g_ma));
    a = es.eigenvectors();

    return a;
}

MatrixXd MD_OC::Eigenvalue_Even()
{
    MatrixXd b;

    SelfAdjointEigenSolver<MatrixXd> es(Matrix_Even(sys, g_ma));
    b = es.eigenvalues();

    return b;
}

MatrixXd MD_OC::Eigenvector_Odd()
{
    MatrixXd a;

    SelfAdjointEigenSolver<MatrixXd> es(Matrix_Odd(sys, g_ma));
    a = es.eigenvectors();

    return a;
}

MatrixXd MD_OC::Eigenvalue_Odd()
{
    MatrixXd b;

    SelfAdjointEigenSolver<MatrixXd> es(Matrix_Odd(sys, g_ma));
    b = es.eigenvalues();

    return b;
}

///////////////////////////////////////////////////////////////////////


void MD_OC::Hamiltonian_N(MatrixXd even, MatrixXd odd)
{
    MatrixXd INT_odd = MatrixXd::Zero(3,3);
    MatrixXd INT_even = MatrixXd::Zero(3,3);
    double Blank = 0;

    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++)
    {
        INT_even(i,j) = -1 * even(i,j) * i; // -\sum_1^\infty \alpha_i \sin{i\phi}
        
        if (i<2)
        {
            INT_odd(i+1,j) = odd(i,j);
        }
    }
    for (int i = 0; i < M ; i++)
    {
        Blank += coup_Arr[i];
    }

    INT_even(1,0) = INT_even(1,0) * -1;
    INT_even(2,0) = INT_even(2,0) * -1;

    MatrixXd c = INT_even.transpose() * INT_odd;
    //cout << INT_even << endl;

    H_N(0, 1) = -c(0, 0);
    H_N(1, 0) = c(0, 0);
    H_N(1, 2) = c(1, 0);
    H_N(2, 1) = -c(1, 0);
}

/////////////////////////////////////////////////////////////////////////


void MD_OC::Hamiltonian_loc(MatrixXd evenEigenval, MatrixXd oddEigenval)
{
    H_loc(0, 0) = evenEigenval(0);
    H_loc(1, 1) = oddEigenval(0);
    H_loc(2, 2) = evenEigenval(1);
}

///////////////////////////////////////////////////////////////////////////////


void MD_OC::CAL_COUP_INT_with_g_arr(double alpha, double k_cutoff)
{
    Tilde_g_calculation_function(alpha,k_cutoff);
    Interact_V(k_cutoff);
    Hamiltonian_N(Eigenvector_Even(), Eigenvector_Odd());
    Hamiltonian_loc(Eigenvalue_Even(),Eigenvalue_Odd());


    cout << "$ H_N value : \n " << H_N << endl;
    cout << "$ H_loc value : \n " << H_loc << endl;
}

///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////

void MD_OC::Dataoutput(double g_ma, double alpha)
{
    std::ofstream outputFile("./");
    std::stringstream gam;
    std::stringstream alp;

    gam << g_ma;
    alp << alpha;

    string INT= "INT_Arr_g";

    INT += gam.str();
    INT += "_a";
    INT += alp.str();
    INT += ".dat";

    outputFile.open(INT);
    for (int i = 0; i < t; i++)
    {
        outputFile << tau_grid[i] << "\t" << INT_Arr[i] << endl;
    }
    outputFile.close();

    string HN= "H_N_g";
    HN += gam.str();
    HN += "_a";
    HN += alp.str();
    HN += ".dat";

    outputFile.open(HN);
    outputFile << H_N << endl;
    outputFile.close();

    string HL= "H_loc_g";
    HL += gam.str();
    HL += "_a";
    HL += alp.str();
    HL += ".dat";

    outputFile.open(HL);
    outputFile << H_loc << endl;
    outputFile.close();

}
/////////////////////////////////////////////////////////////////////////////////

int main()
{
    double beta = 0;
    int grid = 0;

    MD_OC OC(beta,grid);

    double& ref_g_ma = g_ma;
    double& alp = alpha;
    int& syst = sys;
    double k_cutoff = 20;

    syst = 21;

    string input;
    getline(cin, input); // 한 줄을 입력받음

    istringstream iss(input);
    iss >> g_ma >> alpha;

    cout << " ** H_loc, INT_Arr, H_N sending process activates";
    cout << " Value of g_ma : " << g_ma << ", alpha : " << alpha << endl;

    OC.CAL_COUP_INT_with_g_arr(alpha,k_cutoff);
    OC.Dataoutput(g_ma,alpha);

    return 0;
}

