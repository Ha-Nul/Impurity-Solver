#include<iostream>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <vector>
#include <cmath>
#include <OCA_bath.hpp>
#include <chrono>

using namespace std;
using namespace Eigen;

MD_OC MD;

/////////////////////////////////////////////////////////////

int MD_OC::M = MD.mode_grid.size();
int MD_OC::t = MD.tau_grid.size();
vector<double> k_mode(100, 1);
double g_ma = 1;

//////////////////////////////////////////////////////////////

vector<double> INT_Arr(MD.t, 0);
vector<double> Chi_Arr(MD.t, 0);

vector<MatrixXd> T_IN(MD.t,MatrixXd::Zero(3,3));
vector<vector<MatrixXd> > T(MD.t,T_IN);

vector<MatrixXd> Chi_IN(MD.t,MatrixXd::Zero(3,3));
vector<vector<MatrixXd> > Chi_st(MD.t,Chi_IN);

vector<MatrixXd> SELF_E(MD.t, MatrixXd::Zero(3, 3));
MatrixXd MD_OC::H_N = MatrixXd::Zero(3,3);

//////////////////////////////////////////////////////////////
/////////// Array for calculate the time per one for loop cycle /////////

vector<double> G_Arr(MD.M,0);
vector<double> omega_Arr(MD.M,0);

/////////////////////////////////////////////////////////////////////////

vector<double> MD_OC::green(vector<double> tau)
{
    double T = 273;
    vector<int> one_vec(t, 1);
    vector<double> bose_dist(t);

    for (int i = 0; i < t; i++)
    {
        bose_dist[i] = one_vec[i] / (exp(tau_grid[t - 1] * k_mode[i]) - 1);
    }

    vector<double> Test_green(t);

    for (int j = 0; j < t; j++)
    {
        Test_green[j] = ((bose_dist[j] + 1) * exp(-1 * k_mode[j] * tau[j]) + (bose_dist[j]) * exp(k_mode[j] * tau[j]));
    }

    return Test_green;
}

void MD_OC::Tilde_g_calculation_function(double alpha, double k_cutoff)
{

    double nu = pi * k_cutoff / alpha;

    //Initializing block
    /*
    for (int i=0; i < M; i++)
    {
        omega_Arr[i] = 0;
        G_Arr[i] = 0;
    }
    */

    for (int i=0; i < M; i++)
    {
        omega_Arr[i] = k_cutoff * (mode_grid[i]/mode_grid[M-1]);
        G_Arr[i] = sqrt((2 * k_cutoff / (alpha * M)) * (omega_Arr[i] / (1 + pow(nu * omega_Arr[i] / k_cutoff,2))));
        //tilde_g_arr[i] = sqrt( (omega_arr[i] / (1 + pow(nu * omega_arr[i] / k_cutoff,2))));
        //tilde_g_arr[i] = sqrt((2 * k_cutoff / (alpha * omega_arr.size())) * (re_planck_cst * omega_arr[i] / (1 + pow(nu * re_planck_cst * omega_arr[i] / k_cutoff,2))));
    }

    if (alpha == 0)
    {
        for (int i=0; i < M; i++)
        {
            G_Arr[i] = 0;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////

vector<double> MD_OC::Interact_V()
{
    //Initializing block
    /*
    for (int i=0; i < t; i++)
    {
        INT_Arr[i] = 0;
    }
    */

    for (int i = 0; i < t; i++)
    {
        for (int j = 0; j < M ;j++)
        {
            INT_Arr[i] += -pow(G_Arr[j],2) * cosh((tau_grid[i] - tau_grid[t - 1] / 2) * omega_Arr[j])/sinh(tau_grid[t - 1] * omega_Arr[j] / 2); //caution for sign
            //cout << "\t" << j <<" V_arr : " << V_arr[i] << " with tau-beta/2 : " << tau[i] - tau[tau.size()-1]/2 << endl;
        }
    }

    return INT_Arr;
}


////////////////////////////////////////////////////////////////////////////////////

MatrixXd MD_OC::Eigenvector_Even()
{
    MatrixXd a;

    SelfAdjointEigenSolver<MatrixXd> es(Matrix_Even(3, g_ma));
    a = es.eigenvectors();

    return a;
}

MatrixXd MD_OC::Eigenvalue_Even()
{
    MatrixXd b;

    SelfAdjointEigenSolver<MatrixXd> es(Matrix_Even(3, g_ma));
    b = es.eigenvalues();

    return b;
}

MatrixXd MD_OC::Eigenvector_Odd()
{
    MatrixXd a;

    SelfAdjointEigenSolver<MatrixXd> es(Matrix_Odd(3, g_ma));
    a = es.eigenvectors();

    return a;
}

MatrixXd MD_OC::Eigenvalue_Odd()
{
    MatrixXd b;

    SelfAdjointEigenSolver<MatrixXd> es(Matrix_Odd(3, g_ma));
    b = es.eigenvalues();

    return b;
}

///////////////////////////////////////////////////////////////////////


MatrixXd MD_OC::Hamiltonian_N(MatrixXd even, MatrixXd odd)
{
    //cout << "input g value :" << g << endl;
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
        Blank += G_Arr[i];
    }

    INT_even(1,0) = INT_even(1,0) * -1;
    INT_even(2,0) = INT_even(2,0) * -1;

    MatrixXd c = INT_even.transpose() * INT_odd;
    //cout << INT_even << endl;

    H_N(0, 1) = -c(0, 0);
    H_N(1, 0) = c(0, 0);
    H_N(1, 2) = c(1, 0);
    H_N(2, 1) = -c(1, 0);

    cout << H_N << endl;

    return H_N;
}

vector<MatrixXd> MD_OC::Hamiltonian_exp(MatrixXd a, MatrixXd b)
{
    //g_0
    MatrixXd Even = a;
    MatrixXd Odd = b;

    double zeroth = exp(Even(0));
    double first = exp(Odd(0));
    double second = exp(Even(1));

    vector<MatrixXd> array_with_Matrix(t);

    MatrixXd Hamiltonian_exp;

    for (int i = 0; i < t; i++)
    {
        Hamiltonian_exp = MatrixXd::Zero(3, 3);

        Hamiltonian_exp(0, 0) = tau_grid[i] * zeroth;
        Hamiltonian_exp(1, 1) = tau_grid[i] * first;
        Hamiltonian_exp(2, 2) = tau_grid[i] * second;

        array_with_Matrix[i] = Hamiltonian_exp;
    }

    return array_with_Matrix;
}



MatrixXd MD_OC::Hamiltonian_loc(MatrixXd a, MatrixXd b)
{
    MatrixXd Hamiltonian = MatrixXd::Zero(3, 3);

    Hamiltonian(0, 0) = a(0);
    Hamiltonian(1, 1) = b(0);
    Hamiltonian(2, 2) = a(1);

    return Hamiltonian;
}

///////////////////////////////////////////////////////////////////////////////


void MD_OC::CAL_COUP_INT_with_g_arr(double alpha, double k_cutoff)
{
    Tilde_g_calculation_function(alpha,k_cutoff);
    INT_Arr = Interact_V();
    H_N = Hamiltonian_N(Eigenvector_Even(), Eigenvector_Odd());
}


////////////////////////////////////////////////////////////////////////////////

void MD_OC::NCA_self(const MatrixXd& N, const vector<MatrixXd>& Prop, const vector<double>& V)
{
    for (int i = 0; i < t; i++)
    {
        SELF_E[i] = V[i] * (N * Prop[i] * N);
    }
}

void MD_OC::OCA_T(const MatrixXd& N,const vector<MatrixXd>& Prop,const vector<double>& V)
{
    /*
    std::chrono::system_clock::time_point start= std::chrono::system_clock::now();
    cout << "\t" << "T_matrix Calculation Starts" << endl;
    */
    for (int n=0; n<t; n++) for (int m=0; m<=n; m++)
    {
        T[n][m] = N * Prop[n-m] * N * Prop[m] * N;
    }
    /*
    std::chrono::system_clock::time_point sec = std::chrono::system_clock::now();
    std::chrono::duration<double> microseconds = std::chrono::duration_cast<std::chrono::milliseconds>(sec-start);
    cout << "\t" << "Calculation ends : " << microseconds.count() << "[sec]" << endl;
    cout << "-----------------------------" << endl;
    */
}

void MD_OC::OCA_self(const vector<MatrixXd>& Prop)
{
    MatrixXd Stmp;
    int count = 0;

    for (int i = 0; i < t; i++)
    {
        Stmp = MatrixXd::Zero(3, 3);
         for (int n = 0; n <= i; n++) for (int m = 0; m <= n; m++)
        {
            /*
            count += 1;
            std::chrono::system_clock::time_point start= std::chrono::system_clock::now();
            */
            //cout << "\t" << "\t" <<  "For loop count : " << count  << endl;
            /********************main code**************************/
            Stmp += H_N * Prop[i-n] * T[n][m] * INT_Arr[i-m] * INT_Arr[n];                                                                                                                                                                                                                                                                                                                                                                                                                                                 Prop[m] * H_N * INT_Arr[i - m] * INT_Arr[n];
            /*******************************************************/
            /*
            std::chrono::system_clock::time_point sec = std::chrono::system_clock::now();
            std::chrono::duration<double> nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(sec-start);
                if (nanoseconds.count() > 1e-5)
                {
                    cout << "***** (" << n << "," << m << ") ***** : " << nanoseconds.count() << "[sec]" << endl;
                    OCA_TIME[count-1] = 1;
                }
            cout << "\t" << "\t" << "Calculation ends : " << nanoseconds.count() << "[sec]" << endl;
            cout << "-----------------------------------------------------" << endl;
            */
            

        }
        SELF_E[i] += pow(Delta_t, 2) * Stmp;
    }
}


void MD_OC::SELF_Energy(vector<MatrixXd> &Prop)
{
    //cout << "Self_E calculation starts" << endl;
    OCA_T(H_N, Prop, INT_Arr);
    NCA_self(H_N, Prop, INT_Arr);
    std::chrono::system_clock::time_point start= std::chrono::system_clock::now();
    cout << "\t" << "OCA calculation Starts" << endl;
    OCA_self(Prop);
    std::chrono::system_clock::time_point sec = std::chrono::system_clock::now();
    std::chrono::duration<double> microseconds = std::chrono::duration_cast<std::chrono::milliseconds>(sec-start);
    cout << "\t" << "Calculation ends : " << microseconds.count() << "[sec]" << endl;
    cout << "-----------------------------" << endl;

    //cout << SELF_E[99] << endl;
}

//////////////////////////////////////////////////////////////////////////////


MatrixXd MD_OC::round_propagator_ite(const MatrixXd& loc, const vector<MatrixXd>& sigma, const vector<MatrixXd>& ite, int n, int boolean)
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



vector<MatrixXd> MD_OC::Propagator(const vector<MatrixXd>& sig, const MatrixXd& loc)
{
    vector<MatrixXd> P_arr(t, MatrixXd::Zero(3, 3));
    vector<MatrixXd> S_arr(t, MatrixXd::Zero(3, 3));

    P_arr[0] = MatrixXd::Identity(3, 3);
    S_arr[0] = MatrixXd::Identity(3, 3);

    MatrixXd sig_form = MatrixXd::Zero(3, 3);
    MatrixXd sig_late = MatrixXd::Zero(3, 3);

    for (int i = 1; i < t; i++)
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

double MD_OC::chemical_poten(MatrixXd prop)
{
    double Trace = prop.trace();
    double lambda = -(1 / tau_grid[t - 1]) * log(Trace);

    return lambda;
}

///////////////////////////////////////////////////////////////////////////////

vector<MatrixXd> MD_OC::Iteration(const int& n)
{
    vector<MatrixXd> Prop(t, MatrixXd::Zero(3, 3));
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
            for (int j = 0; j < t; j++)
            {
                Prop[j](0, 0) = exp(-tau_grid[j] * H_loc[0](0, 0));
                Prop[j](1, 1) = exp(-tau_grid[j] * H_loc[0](1, 1));
                Prop[j](2, 2) = exp(-tau_grid[j] * H_loc[0](2, 2));
            }

            //cout << Prop[99] << endl;


            lambda[0] = chemical_poten(Prop[t - 1]);
            expDtauLambda = exp((tau_grid[1] - tau_grid[0]) * lambda[0]);
            factor = 1.0;


            for (int j = 0; j < t; j++)
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

            lambda[i] = chemical_poten(Prop[t - 1]);

            expDtauLambda = exp((tau_grid[1] - tau_grid[0]) * lambda[i]);
            factor = 1.0;

            for (int j = 0; j < t; j++)
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

void MD_OC::NCA_Chi_sp(vector<MatrixXd> iter)
{
    MatrixXd GELL_1 = MatrixXd::Zero(3, 3);
    GELL_1(0, 1) = 1;
    GELL_1(1, 0) = 1;

    for (int i = 0; i < t; i++)
    {
        Chi_Arr[i] = (iter[t - i - 1] * GELL_1 * iter[i] * GELL_1).trace();
    }
}

void MD_OC::OCA_store(vector<MatrixXd> iter)
{
    MatrixXd GELL_1 = MatrixXd::Zero(3,3);
    GELL_1(0, 1) = 1;
    GELL_1(1, 0) = 1;

    for (int n=0; n<t; n++) for (int m=0; m<=n; m++)
    {
        Chi_st[n][m] = iter[n-m] * H_N * iter[m] * GELL_1;
        //cout << "pair (n,m) is : " <<  "(" << n << "," << m << ")" << "corresponds with" << "(" << n-m << "," << m << ")" << endl;
    }
}

void MD_OC::OCA_Chi_sp(vector<MatrixXd> iter)
{
    for (int i=0; i<t; i++)
    {
        MatrixXd Stmp = MatrixXd::Zero(3, 3);

        for (int n = 0; n <= i; n++) for (int m = i; m < t; m++)
        {
            Stmp += INT_Arr[m-n] * ( Chi_st[t-i-1][m-i] * Chi_st[i][n]);
            //cout << "pair ("<<n<<","<<m<<") is : " << "(" << k-i-1 << "," << m-i << ")"<< " with " << "(" << i << "," << n << ")" << endl;
        }
        Chi_Arr[i] += pow(Delta_t, 2) * Stmp.trace();
    }
}

vector<double> MD_OC::Chi_sp_Function(vector<MatrixXd> ITE)
{
    NCA_Chi_sp(ITE);
    OCA_store(ITE);
    OCA_Chi_sp(ITE);
    
    return Chi_Arr;
    
}
////////////////////////////////////////////////////////////////////////////////////

int main()
{
    MD_OC MD;
    std::chrono::system_clock::time_point P_start= std::chrono::system_clock::now();
    cout << " ## OCA Program begins ##" << endl;
    cout << "-------------------------------" << endl;
    int modeselec = 0;
    /*
    double& taulimit = MD.beta;

    cout << " Set BETA values to calculate : ";
    cin >> taulimit;

    cout << "\n" << "Calculation would be done under " << taulimit << " value";
    */

    //while (modeselec != -1)

    //cout << "< Select mode to run >" << "\n"  << " 1. Prop(G), 2. Chi, 3. beta*Chi " << "\n" << "MODE INDEX : ";
    //cin >> modeselec;

    double alpha = 0;
    double k_cutoff = 20;
    double& ref_g_ma = g_ma;
    
    vector<double> alp_arr(11,0);
    for (int i = 0; i < 11 ; i++)
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
    
    
    vector<double> g_ma_arr(11,0);
    for (int i = 0; i < 11 ; i++)
    {
        if (i==0)
        {
            g_ma_arr[i] = 0.005;
        }
        if (i!=0)
        {
            g_ma_arr[i] = g_ma_arr[i-1] + 0.0005;
        }
    }
    

    for (int al = 0; al < alp_arr.size(); al++)
    {
        //ref_g_ma = g_ma_arr[ga];
        alpha = alp_arr[al];
        ref_g_ma = 0.02;
        /*
        {
            std::ofstream outputFile ("/Users/e2_602_qma/Documents/GitHub/Anaconda/C++_Mac/EXECUTION");

                string name = "OCA_HYB_g_ma";

                std::stringstream gam;
                std::stringstream alp;
                std::stringstream cuof;
                std::stringstream bet;
                std::stringstream gri;

                gam << g_ma;
                alp << alpha;
                cuof << k_cutoff;
                bet << MD.tau_grid[MD.t-1];
                gri << MD.t;

                name += gam.str();
                name += "_ALPHA_";
                name += alp.str();
                name += "_MODE_";
                name += cuof.str();
                name += "_BETA_";
                name += bet.str();
                name += "_GRID_";
                name += gri.str();
                name += ".txt";

                MD.CAL_COUP_INT_with_g_arr(alpha,k_cutoff);
                //vector<MatrixXd> a = MD.Iteration(1);
        
                outputFile.open(name);
                for (int i = 0; i < MD.t; i++)
                {
                    outputFile << MD.tau_grid[i] << "\t" << INT_Arr[i] << endl;
                }
                outputFile.close();
        }
            
            /****************************************************************************/
        
        {

            /****************************G(tau) Calcultaion******************************/
            /*
            for (int i=0; i<1; i++)
            {
                std::ofstream outputFile ("/Users/e2_602_qma/Documents/GitHub/Anaconda/C++_Mac/EXECUTION");

                string name = "OCA_PROP_g_ma_";

                std::stringstream gam;
                std::stringstream alp;
                std::stringstream cuof;
                std::stringstream bet;
                std::stringstream gri;

                gam << g_ma;
                alp << alpha;
                cuof << k_cutoff;
                bet << MD.tau_grid[MD.t-1];
                gri << MD.t;

                name += gam.str();
                name += "_ALPHA_";
                name += alp.str();
                name += "_MODE_";
                name += cuof.str();
                name += "_BETA_";
                name += bet.str();
                name += "_GRID_";
                name += gri.str();
                name += ".txt";

                MD.CAL_COUP_INT_with_g_arr(alpha,k_cutoff);
                vector<MatrixXd> a = MD.Iteration(1);
        
                outputFile.open(name);
                for (int i = 0; i < MD.t; i++)
                {
                    outputFile << MD.tau_grid[i] << "\t" << (a[i])(0,0)<< "\t" << (a[i])(0,1) << "\t" << (a[i])(0,2) << "\t"
                    << (a[i])(1,0) << "\t" << (a[i])(1,1) << "\t"  << (a[i])(1,2) << "\t"
                    << (a[i])(2,0) << "\t" << (a[i])(2,1) << "\t" << (a[i])(2,2) << "\t" << endl; //변수 a에 값을 할당 후 벡터 각 요소를 반복문으로 불러옴. 이전에는 a 대신 함수를 반복해서 호출하는 방법을 썼는데 그래서 계산 시간이 오래 걸림.
                    cout << setprecision(16);
                }
                outputFile.close();
            }
            /****************************************************************************/
        }
        

        {
            //cout << "input g_ma value : ";
            //cin >> ref_g_ma;

            /********************Chi(\tau) Calculation****************************/
            /*
            for (int i=0; i<1; i++)
            {
                std::ofstream outputFile ("/Users/e2_602_qma/Documents/GitHub/Anaconda/C++_Mac/EXECUTION");

                string name = "OCA_CHI_g_ma_";
                
                std::stringstream gam;
                std::stringstream alp;
                std::stringstream cuof;
                std::stringstream bet;
                std::stringstream gri;

                gam << g_ma;
                alp << alpha;
                cuof << k_cutoff;
                bet << MD.tau_grid[MD.t-1];
                gri << MD.t;

                name += gam.str();
                name += "_ALPHA_";
                name += alp.str();
                name += "_MODE_";
                name += cuof.str();
                name += "_BETA_";
                name += bet.str();
                name += "_GRID_";
                name += gri.str();
                name += ".txt";

                MD.CAL_COUP_INT_with_g_arr(alpha,k_cutoff);
                vector<MatrixXd> ITER = MD.Iteration(1);
                vector<double> a = MD.Chi_sp_Function(ITER);

                outputFile.open(name);

                for (int j = 0; j < MD.tau_grid.size(); j++)
                {
                    outputFile << MD.tau_grid[j] << "\t" << a[j] << endl;
                }

            outputFile.close();
            
            }
            /*************************************************************************/
        }
        
        //if (modeselec == 3)
            //cout << "input g_ma value : ";
            //cin >> ref_g_ma;
            /********************\beta * Chi(\beta / 2) Calculation****************************/
            for (int i=0; i<1; i++)
            {
                std::ofstream outputFile ("/Users/e2_602_qma/Documents/GitHub/Anaconda/C++_Mac/EXECUTION");

                string name = "OCA_BETATIMES_CHI_GAMMA_";
                
                std::stringstream gam;
                std::stringstream alp;
                std::stringstream cuof;
                std::stringstream bet;
                std::stringstream gri;

                gam << g_ma;
                alp << alpha;
                cuof << k_cutoff;
                bet << MD.tau_grid[MD.t-1];
                gri << MD.t;

                name += gam.str();
                name += "_ALPHA_";
                name += alp.str();
                name += "_MODE_";
                name += cuof.str();
                name += "_BETA_";
                name += bet.str();
                name += "_GRID_";
                name += gri.str();
                name += ".txt";

                MD.CAL_COUP_INT_with_g_arr(alpha,k_cutoff);
                vector<MatrixXd> ITER = MD.Iteration(20);
                vector<double> a = MD.Chi_sp_Function(ITER);

                outputFile.open(name);

                for (int j = 0; j < MD.tau_grid.size(); j++)
                {
                    outputFile << MD.tau_grid[j] << "\t" << MD.tau_grid[MD.t-1] * a[j] << endl;
                }

                outputFile.close();
            }
            /**************************************************************************/

        
        
        std::chrono::system_clock::time_point P_sec = std::chrono::system_clock::now();
        std::chrono::duration<double> seconds = std::chrono::duration_cast<std::chrono::seconds>(P_sec-P_start);
        cout << "## Total Process ends with : " << seconds.count() << "[sec] ##" << endl;
        cout << "-----------------------------" << endl;
    


    }

    //if (modeselec == -1)
    {
        cout << "Program will shut down" << endl;
        //break;
    }

    

    return 0;

}


