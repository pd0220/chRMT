// simulating chiral Random Matrix Theory (chRMT) model

// used headers and libraries
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <math.h>
#include <complex>
#include <random>
#include <algorithm>

// imaginary unit
const std::complex<double> I(0., 1.);

// helper lambdas
// squaring
auto sq = [](auto const &x)
{
    return x * x;
};

// main function
int main(int, char **argv)
{
    // reading user given parameters
    //
    // linear size of matrix
    int N = std::atoi(argv[1]);
    // quark mass parameter
    double mass = std::atof(argv[2]);
    // chemical potential parameter
    double mu = std::atof(argv[3]);
    // number of Metropolis sweeps (MC time)
    int T = std::atoi(argv[4]);
    // number of configurations to discard to decrease autocorrelation
    int tau = std::atoi(argv[5]);
    // number of flavours
    int Nf = std::atoi(argv[6]);

    // mass, squared mass and chemical potential matrices proportional to the identity
    Eigen::MatrixXd massId = mass * Eigen::MatrixXd::Identity(N, N);
    Eigen::MatrixXd massSqId = sq(mass) * Eigen::MatrixXd::Identity(N, N);
    Eigen::MatrixXd muId = mu * Eigen::MatrixXd::Identity(N, N);

    // random number generation with Mersenne Twister
    std::random_device rd{};
    std::mt19937 gen(rd());
    // normal distribution
    std::normal_distribution<double> normalDistr(0., 1. / std::sqrt(2. * N));
    // uniform random from [0, 1]
    std::uniform_real_distribution<double> uniformDistr(0., 1.);
    // random number generator lambdas
    auto RandNormal = [&normalDistr, &gen]()
    {
        return normalDistr(gen);
    };
    auto RandUniform = [&uniformDistr, &gen]()
    {
        return uniformDistr(gen);
    };

    // generating a random complex square matrix with given size
    auto RandMatrix = [&N, &RandNormal]()
    {
        // initialise matrix
        Eigen::MatrixXcd mat(N, N);
        // add entries
        for (int iRow = 0; iRow < N; iRow++)
        {
            for (int iCol = 0; iCol < N; iCol++)
            {
                mat(iRow, iCol) = RandNormal() + I * RandNormal();
            }
        }

        // return random matrix
        return mat;
    };

    // computing the action
    auto Action = [&N, &Nf, &mu](Eigen::MatrixXcd const &W, double const &det)
    {
        return N * W.squaredNorm() - Nf * std::log(det) - N * sq(mu);
    };

    // acceptance rate for the Metropolis algorithm
    auto Rate = [](double const &deltaAction)
    {
        if (deltaAction <= 0)
            return 1.;
        else
            return std::exp(-deltaAction);
    };

    // Fermionic operator
    auto FermionicOperator = [&N, &massId, &muId](Eigen::MatrixXcd const &W, Eigen::MatrixXcd const &WDagger)
    {
        Eigen::MatrixXcd M = Eigen::MatrixXcd(2 * N, 2 * N);
        // add block matrices
        M.block(0, 0, N, N) = massId;
        M.block(0, N, N, N) = I * W + muId;
        M.block(N, 0, N, N) = I * WDagger + muId;
        M.block(N, N, N, N) = massId;

        // return the fermionic operator
        return M;
    };

    // fermion determinant
    auto FermionDeterminant = [&massSqId, &muId](Eigen::MatrixXcd const &W, Eigen::MatrixXcd const &WDagger)
    {
        // return the fermion determinant through a simplified expression for the determinant
        return (massSqId - (I * WDagger + muId) * (I * W + muId)).fullPivLu().determinant();
    };

    // chiral condensate
    auto ChiralCondensate = [](Eigen::MatrixXcd const &MInverse)
    {
        return MInverse.trace();
    };

    // number density
    auto NumberDensity = [&N](Eigen::MatrixXcd const &MInverse)
    {
        return (MInverse.block(0, N, N, N).trace() + MInverse.block(N, 0, N, N).trace());
    };

    // simulations starts here
    //
    // chiral condensate container for reweightong to the original theory from the phase quenched theory
    std::vector<std::complex<double>> sigma(0);
    // number density container for reweighting to the original theory from the phase quenched theory
    std::vector<std::complex<double>> density(0);
    // squared fermion determinant container for reweighting to the original theory from the phase quenched theory
    std::vector<std::complex<double>> fermionDet(0);

    // preallocating memory for the used matrices and initialising ~ avoiding repeated memory allocation
    // random matrix
    Eigen::MatrixXcd W = RandMatrix();
    // random matrix for the next sweep
    Eigen::MatrixXcd WNew(N, N);
    // adjoint of the random matrix
    Eigen::MatrixXcd WDagger = W.adjoint();
    // adjoint of the random matrix for the next step
    Eigen::MatrixXcd WDaggerNew(N, N);
    // fermionic operator matrix ~ M = D + m
    Eigen::MatrixXcd M(2 * N, 2 * N);
    // inverse of fermionic matrix
    Eigen::MatrixXcd MInverse(2 * N, 2 * N);
    // fermion determinant
    std::complex<double> det = FermionDeterminant(W, WDagger);
    // action
    double action = Action(W, std::abs(det));

    // starting sweeps
    for (int t = 0; t < T; t++)
    {
        // Metropolis step
        // new matrix, adjoint matrix, fermion determinant and action
        WNew = W + 0.15 * RandMatrix();
        WDaggerNew = WNew.adjoint();
        std::complex<double> detNew = FermionDeterminant(WNew, WDaggerNew);
        double actionNew = Action(WNew, std::abs(detNew));

        // change in the action
        double deltaAction = actionNew - action;

        // compute rate
        double rate = Rate(deltaAction);
        // generate a uniform random number from [0, 1]
        double r = RandUniform();

        // decide if the random matrix is accepted or not
        if (r < rate)
        {
            // rewrite configuration and precomputed quantitites
            W = WNew;
            WDagger = WDaggerNew;
            det = detNew;
            action = actionNew;
        }

        // measurements
        //
        if ((t % tau) == 0)
        {
            // fermionic operator
            M = FermionicOperator(W, WDagger);
            // inverse of fermionic operator
            MInverse = M.fullPivLu().inverse();
            // reweighting factor
            std::complex<double> reweightingFactor = std::pow(det, Nf) / std::pow(std::abs(det), Nf);
            // prefactor
            double pf = Nf / 2. / N;
            // measuring chiral condensate for reweighting
            sigma.push_back(pf * ChiralCondensate(MInverse) * reweightingFactor);
            // measuring number density for reweighting
            density.push_back((pf * NumberDensity(MInverse) + mu) * reweightingFactor);
            // measuring fermion determinant for reweighting
            fermionDet.push_back(reweightingFactor);
        }
    }

    for (int i = 0; i < static_cast<int>(sigma.size()); i++)
    {
        std::cout << sigma[i].real() << " " << sigma[i].imag() << " "
                  << density[i].real() << " " << density[i].imag() << " "
                  << fermionDet[i].real() << " " << fermionDet[i].imag() << std::endl;
    }

    // calculate averages
    // std::complex<double> sigmaNum = std::accumulate(std::begin(sigma), std::end(sigma), std::complex<double>(0., 0.)) / (double)T;
    // std::complex<double> sigmaDenom = std::accumulate(std::begin(fermionSqDet), std::end(fermionSqDet), std::complex<double>(0., 0.)) / (double)T;
    // std::cout << sigmaNum / sigmaDenom << std::endl;
}
