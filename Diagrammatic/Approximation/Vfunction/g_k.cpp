#include<iostream>
#include <fstream>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <iomanip>
#include <string>
#include <const.h>

using namespace std;
using namespace Eigen;
using namespace dlib;

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

vector<double> mode_arr = linspace(0, 10, 10);
vector<double> tau_arr = linspace(0, 1, 100);
double omega = 1;

void g_k_calculation_function(double alpha, double k_cutoff, double mode)
{
    double nu = pi * k_cutoff / alpha;

    for (int i = 0; i < mode_arr.size(); i++)
    {
        mode_arr[i] = (1 / planck_cst) * sqrt((2 * planck_cst * k_cutoff / alpha * mode) * (planck_cst * mode_arr[i] / 1 + pow(nu * planck_cst * mode_arr[i] / k_cutoff, 2)));
    }
}

vector<double> V_function(vector<double>coupling, vector<double> tau, double omega)
{
    vector<double> hpcos(tau.size(), 0);
    vector<double> hpsin(tau.size(), 0);
    vector<double> V_arr(tau.size(), 0);

    for (int i = 0; i < tau.size(); i++) //sum for tau
    {
        hpcos[i] = cosh((tau[i] - tau[tau.size() - 1] / 2) * omega);
        hpsin[i] = sinh(tau[tau.size() - 1] * omega / 2);
        for (int j = 0; j < coupling.size(); j++) //sum for k_index
        {
            V_arr[i] += (pow(coupling[j], 2) * hpcos[i] / hpsin[i]);
        }
        cout << "this is V_arr " << V_arr[i] << endl;
    }

    return V_arr;
}

int main()
{
    g_calculation_function(1, 10, 10);
    Interact_V(mode_arr, tau_arr, omega);

    return 0;
}
