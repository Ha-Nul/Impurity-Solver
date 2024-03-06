#include<iostream>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <vector>
#include <cmath>
#include <OCA_Function_def.hpp>
#include <chrono>

using namespace std;
using namespace Eigen;

MAIN_DEF MD;

int MAIN_DEF::k = MD.tau_grid.size();
vector<double> k_mode(100, 1);
double g_ma = 1;

double omega = 1;
double velocity = 1;
double cutoff = 1;

vector<double> INT_Arr(MD.k, 0);
vector<double> Chi_sp(MD.k, 0);
vector<MatrixXd> SELF_E(MD.k, MatrixXd::Zero(3, 3));
MatrixXd MAIN_DEF::H_N;

vector<double> MAIN_DEF::green(vector<double> tau)
{
    double T = 273;
    vector<int> one_vec(k, 1);
    vector<double> bose_dist(k);

    for (int i = 0; i < k; i++)
    {
        bose_dist[i] = one_vec[i] / (exp(tau_grid[k - 1] * k_mode[i]) - 1);
    }

    vector<double> Test_green(k);

    for (int j = 0; j < k; j++)
    {
        Test_green[j] = ((bose_dist[j] + 1) * exp(-1 * k_mode[j] * tau[j]) + (bose_dist[j]) * exp(k_mode[j] * tau[j]));
    }

    return Test_green;
}

vector<double> MAIN_DEF::coupling(double v, double g, double W)
{
    vector<double> v_array(k_mode.size(), v);
    vector<double> g_array(k_mode.size(), g);
    vector<double> W_array(k_mode.size(), W);
    vector<double> coupling_array(k_mode.size());

    for (int i = 0; i < k_mode.size(); i++)
    {
        coupling_array[i] = g_array[i] * sqrt(abs(k_mode[i]) * v_array[i] / (1 + pow((abs(k_mode[i]) * v_array[i] / W_array[i]), 2)));
    }

    return coupling_array;
}
////////////////////////////////////////////////////////////////////////////////////

vector<double> MAIN_DEF::Interact_V(vector<double>coupling, vector<double> tau, double omega)
{
    double coupling_const = coupling[0];

    vector<double> hpcos(tau.size(), 0);
    vector<double> hpsin(tau.size(), 0);
    vector<double> coupling_arr(tau.size(), coupling_const * coupling_const);
    vector<double> V_arr(tau.size(), 0);

    for (int i = 0; i < tau.size(); i++)
    {
        hpcos[i] = cosh(tau[i] - tau[tau.size() - 1] / 2) * omega;
        hpsin[i] = sinh(tau[tau.size() - 1] * omega / 2);
        V_arr[i] = (coupling_arr[i] * hpcos[i] / hpsin[i]);

        //cout << "this is V_arr " << V_arr[i] << endl;
    }

    return V_arr;
}

////////////////////////////////////////////////////////////////////////////////////

MatrixXd MAIN_DEF::Eigenvector_Even()
{
    MatrixXd a;

    SelfAdjointEigenSolver<MatrixXd> es(Matrix_Even(3, g_ma));
    a = es.eigenvectors();

    return a;
}

MatrixXd MAIN_DEF::Eigenvalue_Even()
{
    MatrixXd b;

    SelfAdjointEigenSolver<MatrixXd> es(Matrix_Even(3, g_ma));
    b = es.eigenvalues();

    return b;
}

MatrixXd MAIN_DEF::Eigenvector_Odd()
{
    MatrixXd a;

    SelfAdjointEigenSolver<MatrixXd> es(Matrix_Odd(3, g_ma));
    a = es.eigenvectors();

    return a;
}

MatrixXd MAIN_DEF::Eigenvalue_Odd()
{
    MatrixXd b;

    SelfAdjointEigenSolver<MatrixXd> es(Matrix_Odd(3, g_ma));
    b = es.eigenvalues();

    return b;
}

///////////////////////////////////////////////////////////////////////


MatrixXd MAIN_DEF::Hamiltonian_N(MatrixXd even, MatrixXd odd, double g)
{
    cout << "input g value :" << g << endl;
    MatrixXd odd_eigenvec;
    MatrixXd even_eigenvec;

    odd_eigenvec = odd.transpose();
    even_eigenvec = even;

    MatrixXd c;
    c = odd_eigenvec * even_eigenvec;

    MatrixXd d = MatrixXd::Zero(3, 3);

    d(0, 1) = g * c(0, 0);
    d(1, 0) = g * c(0, 0);
    d(1, 2) = g * c(0, 1);
    d(2, 1) = g * c(0, 1);

    return d;
}

vector<MatrixXd> MAIN_DEF::Hamiltonian_exp(MatrixXd a, MatrixXd b)
{
    //g_0 
    MatrixXd Even = a;
    MatrixXd Odd = b;

    double zeroth = exp(Even(0));
    double first = exp(Odd(0));
    double second = exp(Even(1));

    vector<MatrixXd> array_with_Matrix(k);

    MatrixXd Hamiltonian_exp;

    for (int i = 0; i < k; i++)
    {
        Hamiltonian_exp = MatrixXd::Zero(3, 3);

        Hamiltonian_exp(0, 0) = tau_grid[i] * zeroth;
        Hamiltonian_exp(1, 1) = tau_grid[i] * first;
        Hamiltonian_exp(2, 2) = tau_grid[i] * second;

        array_with_Matrix[i] = Hamiltonian_exp;
    }

    return array_with_Matrix;
}



MatrixXd MAIN_DEF::Hamiltonian_loc(MatrixXd a, MatrixXd b)
{
    MatrixXd Hamiltonian = MatrixXd::Zero(3, 3);

    Hamiltonian(0, 0) = a(0);
    Hamiltonian(1, 1) = b(0);
    Hamiltonian(2, 2) = a(1);

    return Hamiltonian;
}

////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////


void MAIN_DEF::CAL_COUP_INT_with_g_arr(double g)
{
    INT_Arr = Interact_V(coupling(velocity, g, cutoff), tau_grid, omega);
    H_N = Hamiltonian_N(Eigenvector_Even(), Eigenvector_Odd(), g);
}


////////////////////////////////////////////////////////////////////////////////


void MAIN_DEF::NCA_self(const MatrixXd& N, const vector<MatrixXd>& Prop, const vector<double>& V)
{
    for (int i = 0; i < k; i++)
    {
        SELF_E[i] = V[i] * (N * Prop[i] * N);
    }
}


void MAIN_DEF::OCA_self(MatrixXd& N, vector<MatrixXd>& Prop, vector<double>& V)
{
    MatrixXd Stmp;

    for (int i = 0; i < k; i++)
    {
        Stmp = MatrixXd::Zero(3, 3);
        for (int n = 0; n <= i; n++) for (int m = 0; m <= n; m++) {
            Stmp += N * Prop[i - n] * N * Prop[n - m] * N * Prop[m] * N * V[i - m] * V[n];
        }
        SELF_E[i] += pow(Delta_t, 2) * Stmp;
    }
}


void MAIN_DEF::SELF_Energy(vector<MatrixXd> Prop)
{
    //cout << "Self_E calculation starts" << endl;
    NCA_self(H_N, Prop, INT_Arr);
    OCA_self(H_N, Prop, INT_Arr);

    //cout << SELF_E[99] << endl;
}


//////////////////////////////////////////////////////////////////////////////


MatrixXd MAIN_DEF::round_propagator_ite(const MatrixXd& loc, const vector<MatrixXd>& sigma, const vector<MatrixXd>& ite, int n, int boolean)
{

    MatrixXd sigsum = MatrixXd::Zero(3, 3);

    if (n == 1)
    {
        sigsum = 0.5 * Delta_t * (sigma[1] * ite[0] + sigma[0] * ite[1]);
    }
    else if (n > 1) {
        for (int i = 0; i < n; i++)
        {
            sigsum += 0.5 * Delta_t * (sigma[n - (i)] * ite[i] + sigma[n - (i + 1)] * ite[i + 1]);

            if (i + 1 == n)
            {
                break;
            }

        }
    }

    //cout << sigsum << endl;

    MatrixXd Bucket = MatrixXd::Zero(3, 3);
    if (boolean == 0)
    {
        Bucket = -loc * ite[n] + sigsum;
    }
    else if (boolean == 1)
    {
        Bucket = sigsum;
    }
    //cout << -loc * ite << endl;
    return Bucket;
}



vector<MatrixXd> MAIN_DEF::Propagator(const vector<MatrixXd>& sig, const MatrixXd& loc)
{
    vector<MatrixXd> P_arr(k, MatrixXd::Zero(3, 3));
    vector<MatrixXd> S_arr(k, MatrixXd::Zero(3, 3));

    P_arr[0] = MatrixXd::Identity(3, 3);
    S_arr[0] = MatrixXd::Identity(3, 3);

    MatrixXd sig_form = MatrixXd::Zero(3, 3);
    MatrixXd sig_late = MatrixXd::Zero(3, 3);

    for (int i = 1; i < k; i++)
    {
        P_arr[1] = P_arr[0];
        sig_late = 0.5 * Delta_t * (0.5 * Delta_t * (sig[1] * P_arr[0] + sig[0] * (P_arr[0] + Delta_t * P_arr[0])));
        P_arr[1] = P_arr[0] - 0.5 * Delta_t * loc * (2 * P_arr[0] + Delta_t * P_arr[0]) + sig_late;
        S_arr[1] = P_arr[1];

        if (i > 1)
        {
            sig_form = round_propagator_ite(loc, sig, P_arr, i - 1, 0);
            S_arr[i] = P_arr[i - 1] + Delta_t * sig_form;

            sig_late = 0.5 * Delta_t * (round_propagator_ite(loc, sig, P_arr, i - 1, 1) + round_propagator_ite(loc, sig, S_arr, i, 1));
            P_arr[i] = P_arr[i - 1] - 0.5 * Delta_t * loc * (2 * P_arr[i - 1] + Delta_t * sig_form) + sig_late;

        }
    }

    return P_arr;
}

/////////////////////////////////////////////////////////////////////////////

double MAIN_DEF::chemical_poten(MatrixXd prop)
{
    double Trace = prop.trace();
    double lambda = -(1 / tau_grid[k - 1]) * log(Trace);

    return lambda;
}

///////////////////////////////////////////////////////////////////////////////

vector<MatrixXd> MAIN_DEF::Iteration(const int& n)
{
    vector<MatrixXd> Prop(k, MatrixXd::Zero(3, 3));
    Prop[0] = MatrixXd::Identity(3,3);

    vector<MatrixXd> H_loc(n + 1, MatrixXd::Zero(3, 3));
    H_loc[0] = Hamiltonian_loc(Eigenvalue_Even(), Eigenvalue_Odd());

    MatrixXd Iden = MatrixXd::Identity(3, 3);

    vector<double> lambda(n + 1, 0);
    double expDtauLambda;
    double factor;

    for (int i = 0; i <= n; i++)
    {
        if (i == 0)
        {
            for (int j = 0; j < k; j++)
            {
                Prop[j](0, 0) = exp(-tau_grid[j] * H_loc[0](0, 0));
                Prop[j](1, 1) = exp(-tau_grid[j] * H_loc[0](1, 1));
                Prop[j](2, 2) = exp(-tau_grid[j] * H_loc[0](2, 2));
            }

            //cout << Prop[99] << endl;


            lambda[0] = chemical_poten(Prop[k - 1]);
            expDtauLambda = exp((tau_grid[1] - tau_grid[0]) * lambda[0]);
            factor = 1.0;


            for (int j = 0; j < k; j++)
            {
                Prop[j] *= factor;
                factor *= expDtauLambda;
                //cout << Prop[j] << endl;
            }
        }

        else
        {
            std::chrono::system_clock::time_point start= std::chrono::system_clock::now();
            cout << "Iteration " << i << " Starts" << endl;
            H_loc[i] = H_loc[i - 1] - lambda[i - 1] * Iden;
            SELF_Energy(Prop);
            Prop = Propagator(SELF_E, H_loc[i]);

            lambda[i] = chemical_poten(Prop[k - 1]);

            expDtauLambda = exp((tau_grid[1] - tau_grid[0]) * lambda[i]);
            factor = 1.0;

            for (int j = 0; j < k; j++)
            {
                Prop[j] *= factor;
                factor *= expDtauLambda;

                //cout << Prop[j] << endl;
            }
            std::chrono::system_clock::time_point sec = std::chrono::system_clock::now();
            std::chrono::duration<double> microseconds = std::chrono::duration_cast<std::chrono::milliseconds>(sec-start);
            cout << "Process ends in : " << microseconds.count() << "[sec]" << endl;
            cout << "-----------------------------" << endl;

        }

    }

    return Prop;
}

//////////////////////////////////////////////////////////////////////////////

void MAIN_DEF::NCA_Chi_sp(vector<MatrixXd> iter)
{
    MatrixXd GELL_1 = MatrixXd::Zero(3, 3);
    GELL_1(0, 1) = 1;
    GELL_1(1, 0) = 1;

    for (int i = 0; i < k; i++)
    {
        Chi_sp[i] = (iter[k - i - 1] * GELL_1 * iter[i] * GELL_1).trace();
    }
}

void MAIN_DEF::OCA_Chi_sp(vector<MatrixXd> iter)
{
    MatrixXd GELL_1 = MatrixXd::Zero(3, 3);
    GELL_1(0, 1) = 1;
    GELL_1(1, 0) = 1;

    for (int i = 0; i < k; i++)
    {
        MatrixXd Stmp = MatrixXd::Zero(3, 3);

        for (int n = 0; n <= i; n++) for (int m = i; m < k; m++)
        {
            Stmp += INT_Arr[m - n] * iter[k - m - 1] * H_N * iter[m - i] * GELL_1 * iter[i - n] * H_N * iter[n] * GELL_1;
        }

        Chi_sp[i] += pow(Delta_t, 2) * Stmp.trace();
    }
}

vector<double> MAIN_DEF::Chi_sp_Function(vector<MatrixXd> ITE)
{
    NCA_Chi_sp(ITE);
    OCA_Chi_sp(ITE);
    
    return Chi_sp;
    
}
////////////////////////////////////////////////////////////////////////////////////

int main()
{
    MAIN_DEF MD;

    std::chrono::system_clock::time_point P_start= std::chrono::system_clock::now();
    cout << " ## Program begins ##" << endl;
    cout << "-------------------------------" << endl;

    vector<double> g_array(25, 0);
    for (int j = 1; j < 25; ++j)
    {
        if (j < 21)
        {
            g_array[j] = (g_array[j - 1] + 0.05);
        }

        else
        {
            g_array[j] = g_array[j - 1] + 1;
        }
    }

    for (int m = 0; m < 21; m++)
    {
        g_array[m] = g_array[m] * g_array[m];
    }
    
    for (int i = 0; i < g_array.size(); i++)
    {

        MD.CAL_COUP_INT_with_g_arr(g_array[i]);
        vector<MatrixXd> ITER = MD.Iteration(3);
        vector<double> a = MD.Chi_sp_Function(ITER);
        
        std::ofstream outputFile;

        //string name = "20240111_Trap_beta_0_4_g_";
        string name = "OCATEST";
        //std::stringstream back;
        //back << g_array[k];

        //name += back.str();
        name += ".txt";

        outputFile.open(name);
        /*
        for (int i = 0; i < a.size(); i++)
        {     
            //cout << (a[i])[0][0] << (a[i])[0][1] << endl;
            outputFile << MD.tau_grid[i] << "\t" << (a[i])(0,0)<< "\t" << (a[i])(0,1) << "\t" << (a[i])(0,2) << "\t" 
            << (a[i])(1,0) << "\t" << (a[i])(1,1) << "\t"  << (a[i])(1,2) << "\t" 
            << (a[i])(2,0) << "\t" << (a[i])(2,1) << "\t" << (a[i])(2,2) << "\t" << endl; //변수 a에 값을 할당 후 벡터 각 요소를 반복문으로 불러옴. 이전에는 a 대신 함수를 반복해서 호출하는 방법을 썼는데 그래서 계산 시간이 오래 걸림.
            cout << setprecision(16);  
        }
        */

        //vector<double> a = test.Interact_V(test.coupling(velocity,g_array[k],cutoff),test.grid,omega);
        
        for (int j = 0; j < a.size(); j++)
        {
            cout << a[j] << endl;
            outputFile << MD.tau_grid[j] << "\t" << a[j] << endl;
        }
        

        outputFile.close();
    }
    std::chrono::system_clock::time_point P_sec = std::chrono::system_clock::now();
    std::chrono::duration<double> seconds = std::chrono::duration_cast<std::chrono::seconds>(P_sec-P_start);
    cout << "## Total Process ends with : " << seconds.count() << "[sec] ##" << endl;
    cout << "-----------------------------" << endl;
    

    return 0;

}
