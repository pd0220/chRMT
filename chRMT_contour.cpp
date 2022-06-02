// simulating chiral Random Matrix Theory (chRMT) model
// with constant imaginary integration contour deformation

// used headers and libraries
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
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
    // countour deformation parameters
    double k1 = std::atof(argv[7]);
    double k2 = std::atof(argv[8]);
    // identity matrix
    Eigen::MatrixXcd id = Eigen::MatrixXcd::Identity(N, N);

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

    // generating a random real square matrix with given size
    auto RandMatrix = [&N, &RandNormal]()
    {
        // initialise matrix
        Eigen::MatrixXcd mat(N, N);
        // add entries
        for (int iRow = 0; iRow < N; iRow++)
        {
            for (int iCol = 0; iCol < N; iCol++)
            {
                mat(iRow, iCol) = RandNormal();
            }
        }

        // return random matrix
        return mat;
    };

    // computing the "bosonic" action; i.e. Gaussian part
    auto ActionBosonic = [&N](Eigen::MatrixXcd const &alpha, Eigen::MatrixXcd const &beta)
    {
        // N tr(XY)
        return (double)N * (alpha.array().square().sum() + beta.array().square().sum());
    };

    // computing the "fermionic" action; i.e. fermionic determinant part
    auto ActionFermionic = [&Nf](auto const &det)
    {
        // -Nf logdet
        return -(double)Nf * std::log(det);
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
    auto FermionicOperator = [&N, &massId, &muId](Eigen::MatrixXcd const &X, Eigen::MatrixXcd const &Y)
    {
        Eigen::MatrixXcd M = Eigen::MatrixXcd(2 * N, 2 * N);
        // add block matrices
        M.block(0, 0, N, N) = massId;
        M.block(0, N, N, N) = I * X + muId;
        M.block(N, 0, N, N) = I * Y + muId;
        M.block(N, N, N, N) = massId;

        // return the fermionic operator
        return M;
    };

    // Dirac operator
    auto DiracOperator = [&N, &muId](Eigen::MatrixXcd const &X, Eigen::MatrixXcd const &Y)
    {
        Eigen::MatrixXcd D = Eigen::MatrixXcd(2 * N, 2 * N);
        // add block matrices
        D.block(0, 0, N, N) = Eigen::MatrixXcd::Zero(N, N);
        D.block(0, N, N, N) = I * X + muId;
        D.block(N, 0, N, N) = I * Y + muId;
        D.block(N, N, N, N) = Eigen::MatrixXcd::Zero(N, N);

        // return Dirac operator
        return D;
    };

    // fermion determinant
    auto FermionDeterminant = [&massSqId, &muId](Eigen::MatrixXcd const &X, Eigen::MatrixXcd const &Y)
    {
        // return the fermion determinant through a simplified expression for the determinant
        return (massSqId - (I * Y + muId) * (I * X + muId)).fullPivLu().determinant();
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
    // fermion determinant container for reweighting to the original theory from the phase quenched theory
    std::vector<std::complex<double>> fermionDet(0);
    // Dirac operator eigenvalue container (no reweighting ~ not expectation value)
    // std::vector<std::complex<double>> DiracEigen(0);

    // preallocating memory for the used matrices and initialising ~ avoiding repeated memory allocation
    // random matrices
    Eigen::MatrixXcd alpha = RandMatrix() + I * k1 * id;
    Eigen::MatrixXcd beta = RandMatrix() + I * k2 * id;
    // random matrices for the next sweep
    Eigen::MatrixXcd alphaNew(N, N);
    Eigen::MatrixXcd betaNew(N, N);
    // fermionic operator matrix ~ M = D + m
    Eigen::MatrixXcd M(2 * N, 2 * N);
    // Dirac operator matrix ~ D
    Eigen::MatrixXcd D(2 * N, 2 * N);
    // inverse of fermionic matrix
    Eigen::MatrixXcd MInverse(2 * N, 2 * N);
    // fermion determinant
    std::complex<double> det = FermionDeterminant(alpha + I * beta, alpha.transpose() - I * beta.transpose());
    // bosonic action
    std::complex<double> bosonic = ActionBosonic(alpha, beta);
    // fermionic action for the simulated theory
    double fermionic = ActionFermionic(std::abs(det));

    // starting sweeps
    for (int t = 0; t < T; t++)
    {
        // Metropolis step
        // new matrix, adjoint matrix, fermion determinant and action
        alphaNew = alpha + 0.15 * RandMatrix();
        betaNew = beta + 0.15 * RandMatrix();
        std::complex<double> detNew = FermionDeterminant(alphaNew + I * betaNew, alphaNew.transpose() - I * betaNew.transpose());
        std::complex<double> bosonicNew = ActionBosonic(alphaNew, betaNew);
        double fermionicNew = ActionFermionic(std::abs(detNew));

        // change in the action of the simulated theory (no sign problem)
        double deltaAction = fermionicNew + bosonicNew.real() - fermionic - bosonic.real();

        // compute rate
        double rate = Rate(deltaAction);
        // generate a uniform random number from [0, 1]
        double r = RandUniform();

        // decide if the random matrix is accepted or not
        if (r < rate)
        {
            // rewrite configuration and precomputed quantitites
            alpha = alphaNew;
            beta = betaNew;
            det = detNew;
            bosonic = bosonicNew;
            fermionic = fermionicNew;
        }

        // measurements
        //
        if ((t % tau) == 0)
        {
            // reweighting factor
            std::complex<double> reweightingFactor = std::pow(det / std::abs(det), Nf) * std::exp(-I * bosonic.imag());
            // fermionic operator
            M = FermionicOperator(alpha + I * beta, alpha.transpose() - I * beta.transpose());
            // inverse of fermionic operator
            MInverse = M.fullPivLu().inverse();
            // prefactor
            double pf = Nf / 2. / N;
            // measuring chiral condensate for reweighting
            sigma.push_back(pf * ChiralCondensate(MInverse) * reweightingFactor);
            // measuring number density for reweighting
            density.push_back((pf * NumberDensity(MInverse) + mu) * reweightingFactor);
            // measuring reweighting factor
            fermionDet.push_back(reweightingFactor);

            /*
            // Dirac operator
            D = DiracOperator(alpha + I * beta, alpha.transpose() - I * beta.transpose());
            // computing eigenvalues
            Eigen::ComplexEigenSolver<Eigen::MatrixXcd> eigensolver(D);
            // check for succes
            if (eigensolver.info() != Eigen::Success) abort();
            // saving eigenvalues
            for(int i = 0; i < 2 * N; i++)
            {
                DiracEigen.push_back(eigensolver.eigenvalues()[i]);
            }
            */
        }
    }

    // writing results to screen
    for (int i = 0; i < static_cast<int>(fermionDet.size()); i++)
    {
        // std::cout << sigma[i].real() << " " << sigma[i].imag() << " "
        //           << density[i].real() << " " << density[i].imag() << " "
        //           << fermionDet[i].real() << " " << fermionDet[i].imag() << std::endl;
    }

    std::complex<double> meanFermionDet = std::accumulate(fermionDet.begin(), fermionDet.end(), 0. + I * 0.) / static_cast<double>(fermionDet.size());
    std::cout << k1 << " " << k2 << " " << meanFermionDet.real() << " " << meanFermionDet.imag() << std::endl;
    
    /*
    for (int i = 0; i < static_cast<int>(DiracEigen.size()); i++)
    {
        std::cout << DiracEigen[i].real() << " " << DiracEigen[i].imag() << std::endl;
    }
    */
}
