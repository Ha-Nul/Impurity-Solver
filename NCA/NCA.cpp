#include<iostream>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <vector>
#include <cmath>
#include <firstheader.hpp>

using namespace std;
using namespace Eigen;

const double g = 0;
vector<double> k_grid(10,1);
double gamma = 1;

////////////////////////////////////////////////////////////////////////

vector<double> Testing::green(double tau)
{
    double T = 273;
    vector<int> one_vec(k,1); // 원소는 1, 길이는 n 짜리 배열..
    vector<double> bose_dist(k);

    for (int i = 0; i < k; i++)
    {
        bose_dist[i]=one_vec[i]/(exp(2 * k_grid[i])-1);
    }

    vector<double> Test_green(k);

    for (int j = 0; j < k_grid.size(); j++)
    {
        Test_green[j] = ((bose_dist[j] + 1)*exp(-1 * k_grid[j] * tau) + (bose_dist[j])*exp(k_grid[j] * tau));
    }


    return Test_green;
}

vector<double> Testing::coupling(double v, double g, double W)
{
    vector<double> v_array(k,v);
    vector<double> g_array(k,g);
    vector<double> W_array(k,W);
    vector<double> coupling_array(k);

    for (int i = 0; i < k_grid.size() ; i++)
    {
        coupling_array[i] = g_array[i] * sqrt(abs(k_grid[i]) * v_array[i]/(1 + pow((abs(k_grid[i]) * v_array[i]/W_array[i]),2)));
        //cout << "this is coupling " << coupling_array[i] << endl;
    }
    
    return coupling_array;
}

//이 함수도 해결해야 함. 
vector<double> Testing::Interact(vector<double> coupling, vector<double> tau)
{
    MatrixXd blank_matrix = MatrixXd::Zero(k,k_grid.size());
    vector<double> blank_factor(k);

    for (int i = 0; i < k; i++){
        double t = tau[i];
        for(int j = 0; j < k_grid.size(); j++)
        {
            blank_matrix(i,j)= (coupling[j] *coupling[j]) * green(t)[j];
        }
        blank_factor[i] = blank_matrix.sum();
        blank_matrix = MatrixXd::Zero(k,k);
    }

    return blank_factor;
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
        hpcos[i] = cosh(tau[i]-tau[tau.size()-1])*omega;
        hpsin[i] = sinh(tau[tau.size()-1] * omega/2);
        V_arr[i] = coupling_arr[i] * hpcos[i] / hpsin[i];

        //cout << "this is V_arr " << V_arr[i] << endl;
    }

    return V_arr;
}

////////////////////////////////////////////////////////////////////////////////////

MatrixXd Testing::Eigenvector_Even()
{
	MatrixXd a;

	SelfAdjointEigenSolver<MatrixXd> es(Matrix_Even(3,gamma));
	a = es.eigenvectors();

	return a;
}

MatrixXd Testing::Eigenvalue_Even()
{
	MatrixXd b;

	SelfAdjointEigenSolver<MatrixXd> es(Matrix_Even(3,gamma));
	b = es.eigenvalues();

	return b;
}

MatrixXd Testing::Eigenvector_Odd()
{
	MatrixXd a;

	SelfAdjointEigenSolver<MatrixXd> es(Matrix_Odd(3,gamma));
	a = es.eigenvectors();

	return a;
}

MatrixXd Testing::Eigenvalue_Odd()
{
	MatrixXd b;

	SelfAdjointEigenSolver<MatrixXd> es(Matrix_Odd(3,gamma));
	b = es.eigenvalues();

	return b;
}


//2*2기준으로 바꾸면 N이 이상해짐. 어떻게 해결해야 할까 교수님하고 애기해봐야겠음.

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



//2023.10.04 시간 배열을 받는 함수로 고치는 중 ㅁㄴㅇㄹㅁㄴㅇㄹㅁㄴㅇㄹ
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

        array_with_Matrix[i] = Hamiltonian_exp; // 어떻게 넣어야 하는가.... 지피티가 알려준대로 넣어봄 일단.
    }

    return array_with_Matrix;
}



MatrixXd Testing::Hamiltonian_loc(MatrixXd a, MatrixXd b)
{
    //로컬 해밀토니안 만들 떄 파이썬은 2*2 배열에서 고유값 가져왔는데 얘는 3*3에서 가져와서 고유값 차이가 나는 것 같음.
    //아 로컬해밀토니안 만든다고 했구나
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

    //cout << "this is Sigma" << endl;
    
    for (int i=0; i < k ; i++)
    {   
        Sigarray[i] = 0.5 * V[i] * (Narray[i] * H_exp[i] * Narray[i]);
        
    }
    
    return Sigarray;
}


//////////////////////////////////////////////////////////////////////////////


MatrixXd Testing::round_propagater_ite(const MatrixXd &loc, const vector<MatrixXd> &sigma, const MatrixXd &ite, int n)
{
    //MatrixXd prop_iter_zero = MatrixXd::Identity(n,n);
    MatrixXd sigsum = MatrixXd::Zero(3,3);
    for(int i = n ; i < k; i++)
    {
        sigsum = sigsum + sigma[i];
        //cout << "this is sig" << endl << sigsum << endl << "sig ends" << endl;
    }

    MatrixXd itesum = MatrixXd::Zero(3,3);
    for(int j = 0; j < n; j++)
    {
        itesum = itesum + sigsum * ite;
        //cout << "this is ite" << endl << itesum << endl << "ite ends" << endl;
    }

    MatrixXd Bucket = MatrixXd::Zero(3,3);
    Bucket = -loc * ite + (tau_grid[1]-tau_grid[0]) * itesum;
    //cout << -loc * ite << endl;
    return Bucket;
}



vector<MatrixXd> Testing::Propagator(int n,const vector<MatrixXd> &array, const MatrixXd &loc)
{
    vector<MatrixXd> proparray(k);
    MatrixXd Iden = MatrixXd::Identity(3,3);

    vector<double> coup = coupling(1,g,10);
    vector<double> Int = Interact_V(coup,tau_grid,1);
    //vector<double> Int = Interact(coup,tau_grid);

    MatrixXd H_N = Hamiltonian_N(Eigenvector_Even(),Eigenvector_Odd(),g);
    //vector<MatrixXd> H_e = Hamiltonian_exp(Eigenvalue_Even(),Eigenvalue_Odd());
    vector<MatrixXd> Sig = array;
    /*
    for(int h = 0; h < Sig.size() ; h++)
    {
        cout << "inside Prop" << endl;
        cout << h << Prop[h] << endl;
    }
    */
    proparray[0] = Iden;
    
    for(int i = 1; i < k; i++)
    {
        proparray[i] = proparray[i-1] + (tau_grid[1]-tau_grid[0]) * round_propagater_ite(loc,Sig,proparray[i-1],n);
        //cout << "this is round propagator" << endl << i << endl << round_propagater_ite(H_loc,Sig,proparray[i-1],n) << endl;
    }

    return proparray;
}

/////////////////////////////////////////////////////////////////////////////

double Testing::chemical_poten(MatrixXd prop)
{
    double Trace = prop.trace();
    double lambda = -(1/tau_grid[k-1]) * log(Trace);

    
    //cout << "this is check" << endl;
    //cout << "grid" << endl << tau_grid[k-1] << endl;
    

    return lambda;
}

///////////////////////////////////////////////////////////////////////////////

vector<MatrixXd> Testing::Iteration(const int &n, int testingint)
{
    vector<MatrixXd> Sig;
    vector<MatrixXd> Prop;
    vector<MatrixXd> Prop_zeroth(k,MatrixXd::Identity(3,3));

    vector<double> coup = coupling(1,g,10);
    vector<double> Int = Interact_V(coup,tau_grid,1);
    //vector<double> Int = Interact(coup,tau_grid);

    /*
    for(int j=0; j<k; j++)
            {
                cout << Int[j] << endl;
            }
    */

    MatrixXd H_loc = Hamiltonian_loc(Eigenvalue_Even(),Eigenvalue_Odd());
    MatrixXd Iden = MatrixXd::Identity(3,3);
    double lambda;
    MatrixXd H_N = Hamiltonian_N(Eigenvector_Even(),Eigenvector_Odd(),g);
    vector<MatrixXd> H_e = Hamiltonian_exp(Eigenvalue_Even(),Eigenvalue_Odd());
    
    for(int i = 0; i < testingint; i++)
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
                //cout << "This is Trace" << endl << Prop[j].trace() << endl;
                cout << Prop[j] << endl;
            }

            lambda = chemical_poten(Prop[k-1]);

            for(int j=0; j<k; j++)
            {
                Prop[j] = Prop[j] * exp(tau_grid[j]*(lambda));
                //cout << "This is Trace" << endl << Prop[j].trace() << endl;
                //cout << Prop[j] << endl;
            }
        }
        /*
        else if(i==1)
        {   
            H_loc = Hamiltonian_loc(Eigenvalue_Even(),Eigenvalue_Odd());
            Sig = Sigma(H_N,H_e,Int);
            Prop = Propagator(n,Sig,H_loc);
            lambda = chemical_poten(Prop[k-1]);
            
            cout << "this is " << i << " th Prop" << endl;
            cout << "this is lambda " << lambda << endl;
            
            for(int j=0; j<k; j++)
            {
                Prop[j] = Prop[j] * exp(tau_grid[j]*(lambda));
                //cout << "This is Trace" << endl << Prop[j].trace() << endl;
                cout << Prop[j] << endl;
            }
        }
        */
        else
        {
            //cout << "this is " << i << " th Prop" << endl;

            H_loc = H_loc - lambda * Iden;
            /*
            cout << "//////////////" << endl;
            cout << H_loc << endl;
            cout << "/////////////" << endl;
            */
            Sig = Sigma(H_N,Prop,Int);
            Prop = Propagator(n,Sig,H_loc);
            lambda = chemical_poten(Prop[k-1]);
            
            //cout << "this is lambda" << lambda << endl;

            for(int j=0; j<k; j++)
            {
                Prop[j] = Prop[j] * exp(tau_grid[j]*(lambda));
                //cout << "This is trace" << endl << Prop[j].trace() << endl;
                //cout << Prop[j] << endl;
            }
            
        }
    
    }

    return Prop;
}

//////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
/*
vector<double> Testing::TestingIteration(const int &n, int testingint)
{
    vector<MatrixXd> Sig;
    vector<MatrixXd> Prop;

    vector<double> coup = coupling(1,0.2,10);
    vector<double> Int = Interact(coup,tau_grid);

    vector<double>element1(testingint,0);
    vector<double>element2(testingint,0);
    vector<double>element3(testingint,0);

    MatrixXd H_loc;
    MatrixXd Iden = MatrixXd::Identity(3,3);
    double lambda;
    vector xgrid(testingint,0);
    MatrixXd H_N = Hamiltonian_N(Eigenvector_Even(),Eigenvector_Odd(),0.2);
    vector<MatrixXd> H_e = Hamiltonian_exp(Eigenvalue_Even(),Eigenvalue_Odd());
    
    for(int i = 0; i < testingint; i++)
    {
        if(i==0)
        {   
            H_loc = Hamiltonian_loc(Eigenvalue_Even(),Eigenvalue_Odd());
            Sig = Sigma(H_N,H_e,Int);
            Prop = Propagator(n,Sig,H_loc);
            lambda = chemical_poten(Prop[k-1]);
            for(int j=0; j<k;j++)
            {
            	Prop[j] = Prop[j] * exp(tau_grid[j]*lambda);
            }

        }
        else
        {
            H_loc = H_loc - lambda * Iden;

            Sig = Sigma(H_N,Prop,Int);
            Prop = Propagator(n,Sig,H_loc);
            lambda = chemical_poten(Prop[k-1]);
            for(int j=0; j<k;j++)
            {
            	Prop[j] = Prop[j] * exp(tau_grid[j]*lambda);
            }
        }

        element1[i] = Prop[k-1](0,0);
        element2[i] = Prop[k-1](1,1);
        element3[i] = Prop[k-1](2,2);
        xgrid[i] = i;
    }
}
*/
//////////////////////////////////////////////////////////////////////////////

vector<double> Testing::Chi_sp(const int &weight, int iteration)
{
    MatrixXd Gellmann_1 = MatrixXd::Zero(3,3);

    Gellmann_1(0,1) = 1;
    Gellmann_1(1,0) = 1;

    vector<double> chi_array(k,0);

    for (int i=0; i<k; i++)
    {
        chi_array[i] = (Iteration(weight,iteration)[k-i-1] * Gellmann_1 * Iteration(weight,iteration)[i] * Gellmann_1).trace();
        cout << chi_array[i] << endl;
    }

    return chi_array;
}


int main()
{
    Testing test;

    test.Chi_sp(0,1);


	return 0;
    
}