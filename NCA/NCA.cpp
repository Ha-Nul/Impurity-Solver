#include<iostream>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <vector>
#include <cmath>
#include <iomanip>
#include <string>

using namespace std;
using namespace Eigen;

vector<double> k_mode(100,1);
double g_ma = 1;
double omega = 1;
double velocity = 1;
double cutoff = 1;

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

        vector<MatrixXd> convolve(const vector<MatrixXd>& Signal,
              const vector<MatrixXd>& Kernel, int n, int i)
        {
        size_t SignalLen = i;
        size_t KernelLen = Kernel.size();
        size_t ResultLen = SignalLen + KernelLen - 1;

        vector<MatrixXd> Result(ResultLen,MatrixXd::Zero(n,n));

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

        vector<double> tau_grid = linspace(0,0.4,400);
        int k = tau_grid.size();

    public:

        vector<double> grid = linspace(0,0.4,400);
        vector<double> green(vector<double> tau);
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

        MatrixXd round_propagater_ite(const MatrixXd &loc, const vector<MatrixXd> &sigma, const vector<MatrixXd> &ite,int weight);
        vector<MatrixXd> Sigma(const MatrixXd &N,const vector<MatrixXd> &H_exp, const vector<double> &V);
        vector<MatrixXd> Propagator(const vector<MatrixXd> &array , const MatrixXd &loc , const double &gvalue);

        double chemical_poten(MatrixXd prop);

        vector<MatrixXd> Iteration(const int &iteration, const double &gvalue);
        vector<double> TestingIteration(const int &n, int testingint);

        vector<double> Chi_sp(int iteration, const double &gvalue);

};


/////////////////////////////////////////////////////////////////////////////////////

vector<double> Testing::green(vector<double> tau)
{
    double T = 273;
    vector<int> one_vec(k,1); // 원소는 1, 길이는 n 짜리 배열..
    vector<double> bose_dist(k);

    for (int i = 0; i < k; i++)
    {
        bose_dist[i]=one_vec[i]/(exp(tau_grid[tau_grid.size()-1] * k_mode[i])-1);
    }

    vector<double> Test_green(k);

    for (int j = 0; j < tau_grid.size(); j++)
    {
        Test_green[j] = ((bose_dist[j] + 1)*exp(-1 * k_mode[j] * tau[j]) + (bose_dist[j])*exp(k_mode[j] * tau[j]));
    }

    return Test_green;
}

vector<double> Testing::coupling(double v, double g, double W)
{
    vector<double> v_array(k_mode.size(),v);
    vector<double> g_array(k_mode.size(),g);
    vector<double> W_array(k_mode.size(),W);
    vector<double> coupling_array(k_mode.size());

    for (int i = 0; i < k_mode.size() ; i++)
    {
        coupling_array[i] = g_array[i] * sqrt(abs(k_mode[i]) * v_array[i]/(1 + pow((abs(k_mode[i]) * v_array[i]/W_array[i]),2)));
    }
    
    return coupling_array;
}
////////////////////////////////////////////////////////////////////////////////////

vector<double> Testing::Interact_V(vector<double>coupling, vector<double> tau, double omega)
{
    double coupling_const = coupling[0];

    vector<double> hpcos(tau.size(),0);
    vector<double> hpsin(tau.size(),0);
    vector<double> coupling_arr(tau.size(),coupling_const * coupling_const);
    vector<double> V_arr(tau.size(),0);

    for (int i = 0; i < tau.size(); i++)
    {
        hpcos[i] = cosh(tau[i]-tau[tau.size()-1]/2)*omega;
        hpsin[i] = sinh(tau[tau.size()-1] * omega/2);
        V_arr[i] = (coupling_arr[i] * hpcos[i] / hpsin[i]);

        //cout << "this is V_arr " << V_arr[i] << endl;
    }

    return V_arr;
}

////////////////////////////////////////////////////////////////////////////////////

MatrixXd Testing::Eigenvector_Even()
{
	MatrixXd a;

	SelfAdjointEigenSolver<MatrixXd> es(Matrix_Even(3,g_ma));
	a = es.eigenvectors();

	return a;
}

MatrixXd Testing::Eigenvalue_Even()
{
	MatrixXd b;

	SelfAdjointEigenSolver<MatrixXd> es(Matrix_Even(3,g_ma));
	b = es.eigenvalues();

	return b;
}

MatrixXd Testing::Eigenvector_Odd()
{
	MatrixXd a;

	SelfAdjointEigenSolver<MatrixXd> es(Matrix_Odd(3,g_ma));
	a = es.eigenvectors();

	return a;
}

MatrixXd Testing::Eigenvalue_Odd()
{
	MatrixXd b;

	SelfAdjointEigenSolver<MatrixXd> es(Matrix_Odd(3,g_ma));
	b = es.eigenvalues();

	return b;
}

///////////////////////////////////////////////////////////////////////


MatrixXd Testing::Hamiltonian_N(MatrixXd even, MatrixXd odd, double g)
{
    MatrixXd odd_eigenvec;
    MatrixXd even_eigenvec;

    odd_eigenvec = odd.transpose();
    even_eigenvec = even;

    MatrixXd c;
    c = odd_eigenvec * even_eigenvec;

    MatrixXd d = MatrixXd::Zero(3,3);

    d(0,1) = g * c(0,0);
    d(1,0) = g * c(0,0);
    d(1,2) = g * c(0,1);
    d(2,1) = g * c(0,1);

    return d;
}

vector<MatrixXd> Testing::Hamiltonian_exp(MatrixXd a, MatrixXd b)
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
        Hamiltonian_exp = MatrixXd::Zero(3,3);

        Hamiltonian_exp(0,0) = tau_grid[i] * zeroth;
        Hamiltonian_exp(1,1) = tau_grid[i] * first;
        Hamiltonian_exp(2,2) = tau_grid[i] * second;

        array_with_Matrix[i] = Hamiltonian_exp;
    }

    return array_with_Matrix;
}



MatrixXd Testing::Hamiltonian_loc(MatrixXd a, MatrixXd b)
{
    MatrixXd Hamiltonian = MatrixXd::Zero(3,3);

    Hamiltonian(0,0) = a(0);
    Hamiltonian(1,1) = b(0);
    Hamiltonian(2,2) = a(1);

    return Hamiltonian;
}

MatrixXd Testing::Hamiltonian_loc_ite(MatrixXd a, MatrixXd b, const double &lambda)
{
    MatrixXd Hamiltonian = MatrixXd::Zero(3,3);

    Hamiltonian(0,0) = a(0)-lambda;
    Hamiltonian(1,1) = b(0)-lambda;
    Hamiltonian(2,2) = a(1)-lambda;

    return Hamiltonian;
}

////////////////////////////////////////////////////////////////////////////////


vector<MatrixXd> Testing::Sigma(const MatrixXd &N,const vector<MatrixXd> &H_exp, const vector<double> &V)
{

    vector<MatrixXd> Narray(k,N);
    vector<MatrixXd> Sigarray(k);
    
    for (int i=0; i < k ; i++)
    {   
        Sigarray[i] = 0.5 * V[i] * (Narray[i] * H_exp[i] * Narray[i]);
    }
    
    return Sigarray;
}


//////////////////////////////////////////////////////////////////////////////


MatrixXd Testing::round_propagater_ite(const MatrixXd &loc, const vector<MatrixXd> &sigma, const vector<MatrixXd> &ite, int n)
{   

    MatrixXd sigsum = MatrixXd::Zero(3,3);
    
    if (n == 1)
    {
        sigsum = sigma[1]*ite[0] + sigma[0]*ite[1];
    }
    else if (n > 1){
        for (int i = 0 ; i < n ; i++)
        {
            sigsum += 0.5 * (sigma[n-(i)] * ite[i] + sigma[n-(i+1)] * ite[i+1]);

            if (i+1 == n)
            {
                break;
            }

        }
    }

    //cout << sigsum << endl;

    MatrixXd Bucket = MatrixXd::Zero(3,3);
    Bucket = -loc * ite[n] + (tau_grid[1]-tau_grid[0]) * sigsum;
    //cout << -loc * ite << endl;
    return Bucket;
}



vector<MatrixXd> Testing::Propagator(const vector<MatrixXd> &array, const MatrixXd &loc, const double &gvalue)
{
    vector<MatrixXd> Propagator_array(k,MatrixXd::Zero(3,3));
    MatrixXd Propagator_array_zero = MatrixXd::Identity(3,3);

    Propagator_array[0] = Propagator_array_zero;

    MatrixXd Sigma_former = MatrixXd::Zero(3,3);
    MatrixXd Sigma_later = MatrixXd::Zero(3,3);
    double Delta_tau = tau_grid[1]-tau_grid[0];

    vector<double> coupling_g = coupling(velocity,gvalue,cutoff);
    vector<double> Vfunction = Interact_V(coupling_g,tau_grid,omega);
    vector<MatrixXd> Sigma_function = array;
    MatrixXd N_matrix = Hamiltonian_N(Eigenvector_Even(),Eigenvector_Odd(),gvalue);

    for (int i=1; i < k; i++)
    {
        Propagator_array[1] = Propagator_array[0] + Delta_tau * round_propagater_ite(loc,Sigma_function,Propagator_array,0);

        if (i > 1)
        {
            Sigma_former = round_propagater_ite(loc,Sigma_function,Propagator_array,i-1);
            Propagator_array[i] = Propagator_array[i-1] + Delta_tau * Sigma_former;

            Sigma_later = round_propagater_ite(loc,Sigma_function,Propagator_array,i);
            Propagator_array[i] = Propagator_array[i-1] + Delta_tau * 0.5 * (Sigma_former + Sigma_later);

        }

    
    }

    return Propagator_array;
}

/////////////////////////////////////////////////////////////////////////////

double Testing::chemical_poten(MatrixXd prop)
{
    double Trace = prop.trace();
    double lambda = -(1/tau_grid[k-1]) * log(Trace);
    
    return lambda;
}

///////////////////////////////////////////////////////////////////////////////

vector<MatrixXd> Testing::Iteration(const int &n, const double &gvalue)
{
    vector<MatrixXd> Sig;
    vector<MatrixXd> Prop;
    vector<MatrixXd> Prop_zeroth(k,MatrixXd::Identity(3,3));
    vector<double> coup = coupling(velocity,gvalue,cutoff);
    vector<double> Int = Interact_V(coup,tau_grid,omega);

    MatrixXd H_loc = Hamiltonian_loc(Eigenvalue_Even(),Eigenvalue_Odd());
    MatrixXd Iden = MatrixXd::Identity(3,3);
    MatrixXd H_N = Hamiltonian_N(Eigenvector_Even(),Eigenvector_Odd(),gvalue);
    vector<MatrixXd> H_e = Hamiltonian_exp(Eigenvalue_Even(),Eigenvalue_Odd());

    double lambda;
    
    for(int i = 0; i <= n; i++)
    {
        if (i==0)
        {   
            Prop = Prop_zeroth;
            //cout << "this is " << i << "th iteration " << endl;
            for(int j=0; j<k; j++)
            {
                Prop[j](0,0) = exp(-tau_grid[j] * Hamiltonian_loc(Eigenvalue_Even(),Eigenvalue_Odd())(0,0));
                Prop[j](1,1) = exp(-tau_grid[j] * Hamiltonian_loc(Eigenvalue_Even(),Eigenvalue_Odd())(1,1));
                Prop[j](2,2) = exp(-tau_grid[j] * Hamiltonian_loc(Eigenvalue_Even(),Eigenvalue_Odd())(2,2));
            }

            lambda = chemical_poten(Prop[k-1]);

            for(int j=0; j<k; j++)
            {
                Prop[j] = Prop[j] * exp(tau_grid[j]*(lambda));

            }
        }
    
        else
        {   
            H_loc = H_loc - lambda * Iden;

            Sig = Sigma(H_N,Prop,Int);
            Prop = Propagator(Sig,H_loc,gvalue);
            lambda = chemical_poten(Prop[k-1]);
            
            //cout << "this is lambda" << lambda << endl;
            
            for(int j=0; j<k; j++)
            {
                Prop[j] = Prop[j] * exp(tau_grid[j]*(lambda));
            }

        }
    
    }

    return Prop;
}

//////////////////////////////////////////////////////////////////////////////

vector<double> Testing::Chi_sp(int iter, const double &gvalue)
{
    MatrixXd Gellmann_1 = MatrixXd::Zero(3,3);

    Gellmann_1(0,1) = 1;
    Gellmann_1(1,0) = 1;

    vector<double> chi_array(k,0);
    vector<MatrixXd> Ite_ra = Iteration(iter,gvalue);

    for (int i=0; i<k; i++)
    {
        chi_array[i] =(Ite_ra[k-i-1] * Gellmann_1 * Ite_ra[i] * Gellmann_1).trace();
        cout << setprecision(16);   
        //cout << chi_array[i] << endl;
    }

    return chi_array;
}

int main()
{

    Testing test;
    vector<double> g_array(21,0);

    for (int j=1; j<21; ++j)
    {
        g_array[j] = (g_array[j-1] + 0.05);
    }

    for (int m=0; m<21; m++)
    {
        g_array[m] = g_array[m] * g_array[m];
    }


    for (int k=0; k<21; k++)
    {
        std::ofstream outputFile;

        string name = "20240111_Trap_beta_0_4_g_";
        std::stringstream back;
        back << g_array[k];

        name += back.str();
        name += ".txt";

        outputFile.open(name);

        vector<double> a = test.Chi_sp(5,g_array[k]);

        for (int i = 0; i < a.size(); i++)
        {     
            cout << a[i] << endl;
            outputFile << test.grid[i] << "\t" << a[i] << endl; //변수 a에 값을 할당 후 벡터 각 요소를 반복문으로 불러옴. 이전에는 a 대신 함수를 반복해서 호출하는 방법을 썼는데 그래서 계산 시간이 오래 걸림.
        }
        outputFile.close();
    }
    
    return 0;

}
