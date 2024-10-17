#include<iostream>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <vector>
#include <cmath>
#include <OCA_bath.hpp>
#include <chrono>
#include <const.h>

using namespace std;
using namespace Eigen;

vector<double> k_mode(30000, 1);
double g_ma = 1;
int siz = 0;
int sys = 1;

///////////////////////////////////////////////////////

MD_OC::MD_OC(double beta, int grid)
    : tau_grid(linspace(0, beta, grid)), t(grid - 1)
{
    mode_grid = linspace(1, 30000, 30000);

    Delta_t = tau_grid[1] - tau_grid[0];

    M = mode_grid.size();
    t = tau_grid.size();
    H_N = MatrixXd::Zero(3, 3);
    H_loc = MatrixXd::Zero(3, 3);

    coup_Arr.resize(M);
    omega_Arr.resize(M);
    INT_Arr.resize(t);

}

MD_OC::~MD_OC()
{
    //blank;
}
///////////////////////

void MD_OC::Tilde_g_calculation_function(double alpha, double k_cutoff)
{
    omega_Arr.resize(M);
    coup_Arr.resize(M);
    double nu = pi * k_cutoff / alpha;

    for (int i = 0; i < M; i++)
    {
        omega_Arr[i] = k_cutoff * (mode_grid[i] / mode_grid[M - 1]);
        coup_Arr[i] = sqrt((2 * k_cutoff / (alpha * M)) * (omega_Arr[i] / (1 + pow(nu * omega_Arr[i] / k_cutoff, 2))));

        //simpson formulae
        //omega_Arr[i] = (mode_grid[i]/mode_grid[M-1]); // fix to x to adjust simpson's rule
        //coup_Arr[i] = sqrt((2 * k_cutoff / (alpha)) * ( k_cutoff * omega_Arr[i] / (1 + pow(nu * omega_Arr[i],2)))); // fix to adjust simpson's rule
    }

    if (alpha == 0)
    {
        for (int i = 0; i < M; i++)
        {
            coup_Arr[i] = 0;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////

void MD_OC::Interact_V(double k_cutoff)
{
    //Initializing block

    for (int i = 0; i < t; i++)
    {
        INT_Arr[i] = 0;
    }


    for (int i = 0; i < t; i++)
    {
        for (int j = 0; j < M; j++)
        {
            INT_Arr[i] += -pow(coup_Arr[j], 2) * cosh((tau_grid[i] - tau_grid[t - 1] / 2) * omega_Arr[j]) / sinh(tau_grid[t - 1] * omega_Arr[j] / 2); //caution for sign
        }
    }

}

////////////////////////////////////////////////////////////////////////////////////

MatrixXd MD_OC::Eigenvector_Even()
{
    SelfAdjointEigenSolver<MatrixXd> es(Matrix_Even(sys, g_ma));
    return es.eigenvectors();
}

MatrixXd MD_OC::Eigenvalue_Even()
{
    SelfAdjointEigenSolver<MatrixXd> es(Matrix_Even(sys, g_ma));
    return es.eigenvalues();
}

MatrixXd MD_OC::Eigenvector_Odd()
{
    SelfAdjointEigenSolver<MatrixXd> es(Matrix_Odd(sys, g_ma));
    return es.eigenvectors();
}

MatrixXd MD_OC::Eigenvalue_Odd()
{
    SelfAdjointEigenSolver<MatrixXd> es(Matrix_Odd(sys, g_ma));
    return es.eigenvalues();
}

////////////////////////////////////////////////////////////////////////////////////

void MD_OC::Hamiltonian_N(MatrixXd even, MatrixXd odd)
{
    //cout << "input g value :" << g << endl;
    MatrixXd INT_odd = MatrixXd::Zero(siz, siz);
    MatrixXd INT_even = MatrixXd::Zero(siz, siz);

    //H_N initialize
    H_N = MatrixXd::Zero(siz, siz);

    //cout << "initialized check" << endl;
    //cout << H_N << "\n" << endl;

    for (int i = 0; i < siz; i++) for (int j = 0; j < siz; j++)
    {
        INT_even(i, j) = -1 * even(i, j) * i; // -\sum_1^\infty \alpha_i \sin{i\phi}

        if (i < siz - 1)
        {
            INT_odd(i + 1, j) = odd(i, j);
        }

    }

    MatrixXd c = INT_even.transpose() * INT_odd;
    //cout << c << endl;

    //stocks matrix elements
    for (int i = 0; i < siz; i++) for (int j = 0; j < siz; j++)
    {
        if (j % 2 != 0 & i % 2 == 0) {
            H_N(i, j) = c(i / 2, j / 2); // even * diff odd
        }
        else if (j % 2 == 0 & i % 2 != 0) {
            H_N(i, j) = c(j / 2, i / 2); // odd * diff even
        }
    }

    //matching sign
    for (int i = 0; i < siz; i++) for (int j = 0; j < siz; j++)
    {
        if (i > j)
        {
            H_N(i, j) = -H_N(i, j);
        }
    }

    //cout << H_N << endl;
}
///////////////////////////////////////////////////////////////////////

MatrixXd Ordercal(MatrixXd even, MatrixXd odd)
{
    MatrixXd Order_param = MatrixXd::Zero(siz, siz);

    cout << "**EVENMAT**" << endl;
    cout << even << endl;

    //** constructing even matrix
    MatrixXd eve_0 = even;//MatrixXd::Zero(sys,sys);
    for (int i = 0; i < sys; i++) for (int j = 0; j < sys; j++)
    {
        if (i == 0) {
            eve_0(i, j) = (1 / sqrt(2)) * even(i, j);
        }
        else {
            eve_0(i, j) = even(i, j);
        }
    }

    //Even OffDiagonal construct
    MatrixXd eve_1 = MatrixXd::Zero(sys + 1, sys + 1);
    MatrixXd eve_2 = MatrixXd::Zero(sys + 1, sys + 1);

    for (int i = 0; i < sys + 1; i++) for (int j = 0; j < sys + 1; j++)
    {
        if (i > 0 && j < sys) {
            if (i > 1) {
                eve_1(i, j) = 0.5 * eve_0(i - 1, j);
            }
            else {
                eve_1(i, j) = eve_0(i - 1, j);
            }
        }
        if (j < sys && i < sys) {
            eve_2(i, j) = eve_0(i, j);
        }
    }
    //Activate to change the direction of groundstate eigenvector
    for (int i = 0; i < sys + 1; i++) {
        eve_1(i, 0) = -eve_1(i, 0);
        eve_2(i, 0) = -eve_2(i, 0);
    }


    MatrixXd eve_off = eve_1.transpose() * eve_2;
    //cout << "EVEOFF" << endl;
    //cout << eve_off << endl;

    //cout << "\t" << "<Mateven 1>" << endl;
    //cout << eve_1.transpose() << endl;
    /*
    for (int i = 0; i < sys+1; i++) for (int j = 0; j< sys+1; j++)
    {
        if (j<sys && i<sys)
        {
            eve_2(i,j) = eve_0(i,j);
        }
    }
    */
    //cout << "\t" << "<Mateven 2>" << endl;
    //cout << eve_2 << endl;
    MatrixXd eve_ele = MatrixXd::Zero(sys, sys);

    for (int i = 0; i < sys; i++) for (int j = 0; j < sys; j++)
    {
        if ((i != j) && (i % 2 == 0) && (j % 2 == 0)) {
            eve_ele(i, j) = eve_off(i / 2, j / 2) + eve_off(j / 2, i / 2);
            //eve_ele(j,i) = eve_off(i/2,j/2) + eve_off(j/2,i/2);
        }

        if ((i == j) && (i % 2 == 0) && (j % 2 == 0)) {
            eve_ele(i, j) = 2 * eve_off(i / 2, j / 2);
            //eve_ele(j,i) = eve_off(i/2,j/2) + eve_off(j/2,i/2);
        }
    }

    //cout << "*****EVEELE*****" << endl;
    //cout << eve_ele << endl;
    ///////////// even matrix construction complete ///////////////

    //constructing odd matrix

    cout << "**ODD Eigen matrix**" << endl;
    cout << odd << endl;
        //odd matrix calculation structure design
    MatrixXd odd_ele1 = MatrixXd::Zero(sys, sys);
    MatrixXd odd_ele2 = MatrixXd::Zero(sys, sys);
    for (int i = 0; i < sys; i++) for (int j = 0; j < sys; j++)
    {
        if (i!=0){
            odd_ele1(i,j)=odd(i-1,j);
        }
    }

    for (int i = 0; i < sys; i++) for (int j = 0; j < sys; j++)
    {
        if (i!=0){
            odd_ele1(i,j)=odd(i-1,j);
        }
    }

    for (int i = 0; i < sys; i++) for (int j = 0; j < sys; j++)
    {
        if (i!=(sys-1)){
            odd_ele2(i,j)=odd(i,j);
        }
    }
        //calculation
    MatrixXd odd_ele = MatrixXd::Zero(sys,sys);
    for (int i = 0; i < sys; i++) for (int j = 0; j < sys; j++)
    {
        if ((i == j) && (i % 2 == 1) && (j % 2 == 1)){
            odd_ele(i,j) = (odd_ele1.transpose() * odd)(i/2,j/2);
        }
        else if ((i < j) && (i % 2 == 1) && (j % 2 == 1)){
            odd_ele(i,j) = 0.5 * (odd_ele1.transpose()*odd + odd_ele2.transpose()*odd_ele1)(i,j);
            odd_ele(j,i) = odd_ele(i,j);
        }
    }

    /*
    cout << "ODD_element 1 is : " << endl;
    cout << odd_ele1 << endl;

    cout << "\n";

    cout << "ODD_element 2 is : " << endl;
    cout << odd_ele2 << endl;
    */

    cout << "Structure check" << endl;
    cout << odd_ele << endl;
    ///////////// odd matrix construction complete ///////////////

    for (int i = 0; i < siz; i++) for (int j = 0; j < siz; j++)
    {
        if (i % 2 == 0 && j % 2 == 0)
        {
            Order_param(i, j) = eve_ele(i, j);
        }

        else if (i % 2 != 0 && j != 0)
        {
            Order_param(i, j) = odd_ele(i, j);
        }

    }
    
    return Order_param;

}


void MD_OC::Hamiltonian_loc(MatrixXd a, MatrixXd b)
{
    int siz = a.rows();
    H_loc = MatrixXd::Zero(siz, siz);

    for (int i = 0; i < siz; i++) for (int j = 0; j < siz; j++)
    {
        if (i == j & i % 2 == 0)
        {
            H_loc(i, j) = a(i / 2);
            /*
            if (a(i / 2) > 30)
            {
                H_loc(i, j) = 30;
            }
            */
        }
        if (i == j & i % 2 != 0)
        {
            H_loc(i, j) = b(i / 2);
            /*
            if (b(i / 2) > 30)
            {
                H_loc(i, j) = 30;
            }
            */
        }
    }
}

///////////////////////////////////////////////////////////////////////////////


void MD_OC::CAL_COUP_INT_with_g_arr(double alpha, double k_cutoff)
{
    Tilde_g_calculation_function(alpha, k_cutoff);
    Interact_V(k_cutoff);
    Hamiltonian_N(Eigenvector_Even(), Eigenvector_Odd());
    Hamiltonian_loc(Eigenvalue_Even(), Eigenvalue_Odd());


    cout << "$ H_N value : \n " << H_N << endl;
    cout << "$ H_loc value : \n " << H_loc << endl;

}



int main()
{
    std::chrono::system_clock::time_point P_start = std::chrono::system_clock::now();
    cout << " ## Approx Program begins ##" << endl;
    cout << " Program now set in One-crossing mode " << endl;
    cout << "-------------------------------" << endl;
    /*
    double& taulimit = MD.beta;

    cout << " Set BETA values to calculate : ";
    cin >> taulimit;

    cout << "\n" << "Calculation would be done under " << taulimit << " value";
    */

    //while (modeselec != -1)

    //cout << "< Select mode to run >" << "\n"  << " 1. Prop(G), 2. Chi, 3. beta*Chi " << "\n" << "MODE INDEX : ";
    //cin >> modeselec;
    double beta;
    int grid;

    cout << " * Set beta : ";
    cin >> beta;

    cout << " * Set grid (number of index, not interval count) : ";
    cin >> grid;

    MD_OC MD(beta, grid);

    double alpha = 0.5;
    double k_cutoff = 20;
    double& ref_g_ma = g_ma;

    int& size = siz;
    int& syst = sys;

    size = 3;
    syst = 21;
    /*
    vector<double> alp_arr(21,0);
    for (int i = 0; i < 21 ; i++)
    {
        if (i==0)
        {
            alp_arr[i] = 0;
        }
        if (i!=0)
        {
            alp_arr[i] = alp_arr[i-1] + 0.1;
        }

    }
    */

    vector<double> alp_arr = {5};

    vector<double> g_ma_arr(21, 0);
    for (int i = 0; i < 21; i++)
    {
        if (i == 0)
        {
            g_ma_arr[i] = 0;
        }
        if (i != 0)
        {
            g_ma_arr[i] = g_ma_arr[i - 1] + 0.05;
        }
    }

    vector<double> output(g_ma_arr.size(), 0);


    for (int al = 0; al < alp_arr.size(); al++)
    {
        for (int ga = 0; ga < g_ma_arr.size(); ga++)
        {
            //ref_g_ma = g_ma_arr[ga];
            //alpha = 1;
            alpha = alp_arr[al];
            ref_g_ma = g_ma_arr[ga];

            MD.CAL_COUP_INT_with_g_arr(alpha, k_cutoff);
            MatrixXd Order_param = Ordercal(MD.Eigenvector_Even(), MD.Eigenvector_Odd());

            cout << "Value" << endl;
            cout << Order_param << endl;


            std::ofstream outputFile("./");

            string name = "OCA_COS_GAMMA_";

            std::stringstream gam;
            std::stringstream alp;
            std::stringstream cuof;
            std::stringstream bet;
            std::stringstream gri;
            std::stringstream sizz;
            std::stringstream kmod;

            gam << g_ma;
            alp << alpha;
            cuof << k_cutoff;
            bet << MD.tau_grid[MD.t - 1];
            gri << MD.t;
            sizz << siz;
            kmod << MD.mode_grid.size();

            name += gam.str();
            name += "_ALPHA_";
            name += alp.str();
            name += "_CUTOF_";
            name += cuof.str();
            name += "_MODE_";
            name += kmod.str();
            name += "_BETA_";
            name += bet.str();
            name += "_GRID_";
            name += gri.str();
            name += "_SIZE_";
            name += sizz.str();
            name += ".txt";

            outputFile.open(name);

            for (int i = 0; i < siz; i++) for (int j = 0; j < siz; j++)
            {
                outputFile << Order_param(i, j) << "\t";
            }

            outputFile.close();
        }

    }

}
//}
//}
