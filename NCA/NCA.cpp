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

        vector<double> tau_grid = linspace(0,0.5,400);
        int k = tau_grid.size();

    public:

        vector<double> grid = linspace(0,0.5,400);
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

        MatrixXd round_propagater_ite(const MatrixXd &loc, const vector<MatrixXd> &sigma, const vector<MatrixXd> &ite,int n, int boolean);
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
        Sigarray[i] = V[i] * (Narray[i] * H_exp[i] * Narray[i]);
    }
    
    return Sigarray;
}


//////////////////////////////////////////////////////////////////////////////


MatrixXd Testing::round_propagater_ite(const MatrixXd &loc, const vector<MatrixXd> &sigma, const vector<MatrixXd> &ite, int n, int boolean)
{

    MatrixXd sigsum = MatrixXd::Zero(3,3);
    double dtau = tau_grid[1]-tau_grid[0];
    
    if (n == 1)
    {
        sigsum = 0.5 * dtau * (sigma[1]*ite[0] + sigma[0]*ite[1]);
    }
    else if (n > 1){
        for (int i = 0 ; i < n ; i++)
        {
            sigsum += 0.5 * dtau * (sigma[n-(i)] * ite[i] + sigma[n-(i+1)] * ite[i+1]);

            if (i+1 == n)
            {
                break;
            }

        }
    }

    //cout << sigsum << endl;

    MatrixXd Bucket = MatrixXd::Zero(3,3);
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



vector<MatrixXd> Testing::Propagator(const vector<MatrixXd> &sig, const MatrixXd &loc, const double &gvalue)
{
    vector<MatrixXd> P_arr(k,MatrixXd::Zero(3,3));
    vector<MatrixXd> S_arr(k,MatrixXd::Zero(3,3));

    P_arr[0] = MatrixXd::Identity(3,3);
    S_arr[0] = MatrixXd::Identity(3,3);

    MatrixXd sig_form = MatrixXd::Zero(3,3);
    MatrixXd sig_late = MatrixXd::Zero(3,3);
    double dtau = tau_grid[1]-tau_grid[0];

    vector<double> coupling_g = coupling(velocity,gvalue,cutoff);
    vector<double> Vfunction = Interact_V(coupling_g,tau_grid,omega);
    vector<MatrixXd> self_E = sig;
    MatrixXd N_matrix = Hamiltonian_N(Eigenvector_Even(),Eigenvector_Odd(),gvalue);

    for (int i=1; i < k; i++)
    {
        P_arr[1] = P_arr[0];
        sig_late = 0.5 * dtau * ( 0.5 * dtau * (self_E[1] * P_arr[0] + self_E[0] * (P_arr[0] + dtau * P_arr[0])));
        P_arr[1] = P_arr[0] - 0.5 * dtau * loc * (2 * P_arr[0] + dtau * P_arr[0]) + sig_late;
        S_arr[1] = P_arr[1];

        if (i > 1)
        {
            sig_form = round_propagater_ite(loc,self_E,P_arr,i-1,0);
            S_arr[i] = P_arr[i-1] + dtau * sig_form;

            sig_late = 0.5 * dtau * (round_propagater_ite(loc,self_E,P_arr,i-1,1) + round_propagater_ite(loc,self_E,S_arr,i,1));
            P_arr[i] = P_arr[i-1] - 0.5 * dtau * loc * (2 * P_arr[i-1] + dtau * sig_form) + sig_late;

        }
    }

    return P_arr;
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

    vector<MatrixXd> H_loc(n+1,MatrixXd::Zero(3,3));
        H_loc[0] = Hamiltonian_loc(Eigenvalue_Even(),Eigenvalue_Odd());

    MatrixXd Iden = MatrixXd::Identity(3,3);
    MatrixXd H_N = Hamiltonian_N(Eigenvector_Even(),Eigenvector_Odd(),gvalue);
    
    vector<double> lambda(n+1,0);
    double pre_lambda;
    double expDtauLambda;
    double factor;
    
    for(int i = 0; i <= n; i++)
    {
        if (i==0)
        {
            Prop = Prop_zeroth;
            for(int j=0; j<k; j++)
            {
                Prop[j](0,0) = exp(-tau_grid[j] * Hamiltonian_loc(Eigenvalue_Even(),Eigenvalue_Odd())(0,0));
                Prop[j](1,1) = exp(-tau_grid[j] * Hamiltonian_loc(Eigenvalue_Even(),Eigenvalue_Odd())(1,1));
                Prop[j](2,2) = exp(-tau_grid[j] * Hamiltonian_loc(Eigenvalue_Even(),Eigenvalue_Odd())(2,2));
            }

            lambda[0] = chemical_poten(Prop[k-1]);
            expDtauLambda = exp((tau_grid[1]-tau_grid[0])*lambda[0]);
            factor = 1.0;
            
            
            for(int j=0; j<k; j++)
            {
                Prop[j] *= factor;
                factor *= expDtauLambda;
                //cout << Prop[j].trace() << endl;
            }
        }
    
        else
        {

            //lambda[i-1] = lambda[i-1]/i * pow(-1,i+1) ;

            //cout << i-1 << "th lambda : "<< "\t" << lambda[i-1] << endl;
            //cout << setprecision(16);
            H_loc[i] = H_loc[i-1] - lambda[i-1] * Iden;

            //cout << i << "th H_loc : " << "\n" << H_loc[i] << endl;
            //cout << setprecision(16);

            Sig = Sigma(H_N,Prop,Int);
            Prop = Propagator(Sig,H_loc[i],gvalue);

            //cout << i << "th Prop : " << "\n" << Prop[k-1] << "\n" << "------------------" << endl;
            //cout << setprecision(16);

            lambda[i] = chemical_poten(Prop[k-1]);

            expDtauLambda = exp((tau_grid[1]-tau_grid[0])*lambda[i]);
            factor = 1.0;
            
            //cout << "this is lambda" << lambda << endl;
            
            for(int j=0; j<k; j++)
            {
                Prop[j] *= factor;
                factor *= expDtauLambda;
                //cout << Prop[j].trace() << endl;
            }

            //cout << "----------------------" << endl;

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

    /*
    MatrixXd H_local = test.Hamiltonian_loc(test.Eigenvalue_Even(),test.Eigenvalue_Odd());
    
    std::ofstream outputFile;
    
    string name = "Hamiltonian_local";

    outputFile.open(name);
    outputFile << H_local << endl; //변수 a에 값을 할당 후 벡터 각 요소를 반복문으로 불러옴. 이전에는 a 대신 함수를 반복해서 호출하는 방법을 썼는데 그래서 계산 시간이 오래 걸림.

    outputFile.close();
    //cout << H_local << endl;
    */

    vector<double> g_array(25,0);
    for (int j=1; j<25; ++j)
    {
        if (j<21)
        {
          g_array[j] = (g_array[j-1] + 0.05);
        }

        else
        {
          g_array[j] = g_array[j-1] + 1;
        }
    }

    for (int m=0; m<21; m++)
    {
        g_array[m] = g_array[m] * g_array[m];
    }
    /*
    for (int n=0; n<1; n++)
    {
        std::ofstream outputFile;

        //string name = "20240111_Trap_beta_0_4_g_";
        string name = "N_matrix_beta_2_g_";
        //std::stringstream back;
        //back << g_array[n];

        //name += back.str();
        name += ".txt";

        outputFile.open(name);

        MatrixXd H_N = test.Hamiltonian_N(test.Eigenvector_Even(),test.Eigenvector_Odd(),1);
        outputFile << H_N << endl; //변수 a에 값을 할당 후 벡터 각 요소를 반복문으로 불러옴. 이전에는 a 대신 함수를 반복해서 호출하는 방법을 썼는데 그래서 계산 시간이 오래 걸림.

        outputFile.close();

    }
    */

    /*
    for (int k=0; k<1; k++)
    {
        std::ofstream outputFile;

        //string name = "20240111_Trap_beta_0_4_g_";
        string name = "Refcheck_grid201_Input_Prop_Ite1";
        std::stringstream back;
        //back << '1';//g_array[k];

        name += back.str();
        name += ".txt";

        outputFile.open(name);
        //vector<double> a = test.Interact_V(test.coupling(velocity,g_array[k],cutoff),test.grid,omega);
        vector<MatrixXd> a = test.Iteration(1,1);

        for (int i = 0; i < a.size(); i++)
        {
            //cout << (a[i])[0][0] << (a[i])[0][1] << endl;
            outputFile << test.grid[i] << "\t" << (a[i])(0,0)<< "\t" << (a[i])(0,1) << "\t" << (a[i])(0,2) << "\t"
            << (a[i])(1,0) << "\t" << (a[i])(1,1) << "\t"  << (a[i])(1,2) << "\t"
            << (a[i])(2,0) << "\t" << (a[i])(2,1) << "\t" << (a[i])(2,2) << "\t" << endl; //변수 a에 값을 할당 후 벡터 각 요소를 반복문으로 불러옴. 이전에는 a 대신 함수를 반복해서 호출하는 방법을 썼는데 그래서 계산 시간이 오래 걸림.
            cout << setprecision(16);
        }
        outputFile.close();
    }
    */
    

   //test.Iteration(4,1);
    


   /*
    std::ofstream outputFile;

    string name = "Vfunc";

    name += ".txt";

    outputFile.open(name);
    //vector<double> a = test.Interact_V(test.coupling(velocity,g_array[k],cutoff),test.grid,omega);
    vector<double> arr = test.Interact_V(test.coupling(velocity,1,cutoff),test.grid,omega);

    for (int i = 0; i < arr.size(); i++)
    {
        //cout << (a[i])[0][0] << (a[i])[0][1] << endl;
        outputFile << test.grid[i] << "\t" << arr[i] << endl;
    }
    
    outputFile.close();
    */
    //test.Iteration(10,1);


    for (int k=0; k<g_array.size(); k++)
    {
        std::ofstream outputFile;

        //string name = "20240111_Trap_beta_0_4_g_";
        string name = "COMP_BEAR_BETA05_GRID400_G";
        std::stringstream back;
        back << g_array[k];

        name += back.str();
        name += ".txt";

        outputFile.open(name);
        //vector<double> a = test.Interact_V(test.coupling(velocity,g_array[k],cutoff),test.grid,omega);
        vector<double> a = test.Chi_sp(5,g_array[k]);

        for (int i = 0; i < a.size(); i++)
        {
            //cout << (a[i])[0][0] << (a[i])[0][1] << endl;
            outputFile << test.grid[i] << "\t" << a[i] << endl; //변수 a에 값을 할당 후 벡터 각 요소를 반복문으로 불러옴. 이전에는 a 대신 함수를 반복해서 호출하는 방법을 썼는데 그래서 계산 시간이 오래 걸림.
        }
        outputFile.close();
    
    }
    
    
    return 0;

}
