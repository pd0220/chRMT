// simulating chiral Random Matrix Theory (chRMT) model
// with constant imaginary integration contour deformation
//
// optimisation with AdaDelta

// -----------------------------------------------------------------------------------------------------------------------------------

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

// -----------------------------------------------------------------------------------------------------------------------------------

// imaginary unit
const std::complex<double> I(0., 1.);

// "stepsize" from one matrix to the next
double eta = 0.15;
// optimisation parameters
double g = 0.95;
double eps = 1e-6;

// helper lambdas
// squaring
auto sq = [](auto const &x)
{
    return x * x;
};

// -----------------------------------------------------------------------------------------------------------------------------------

// computing the "bosonic" action; i.e. Gaussian part
auto ActionBosonic(int const &N, Eigen::MatrixXcd const &alpha, Eigen::MatrixXcd const &beta)
{
    // N tr(XY)
    return (double)N * (alpha.array().square().sum() + beta.array().square().sum());
}

// computing the "fermionic" action; i.e. fermionic determinant part
template <typename T>
auto ActionFermionic(int const &Nf, T const &det)
{
    // -Nf logdet
    return -(double)Nf * std::log(det);
}

// -----------------------------------------------------------------------------------------------------------------------------------

// acceptance rate for the Metropolis algorithm
auto Rate(double const &deltaAction)
{
    if (deltaAction <= 0)
        return 1.;
    else
        return std::exp(-deltaAction);
}

// -----------------------------------------------------------------------------------------------------------------------------------

// Fermionic operator
auto FermionicOperator(int const &N, Eigen::MatrixXd const &massId, Eigen::MatrixXd const &muId, Eigen::MatrixXcd const &X, Eigen::MatrixXcd const &Y)
{
    Eigen::MatrixXcd M = Eigen::MatrixXcd(2 * N, 2 * N);
    // add block matrices
    M.block(0, 0, N, N) = massId;
    M.block(0, N, N, N) = I * X + muId;
    M.block(N, 0, N, N) = I * Y + muId;
    M.block(N, N, N, N) = massId;

    // return the fermionic operator
    return M;
}

// -----------------------------------------------------------------------------------------------------------------------------------

// Dirac operator
auto DiracOperator(int const &N, Eigen::MatrixXd const &muId, Eigen::MatrixXcd const &X, Eigen::MatrixXcd const &Y)
{
    Eigen::MatrixXcd D = Eigen::MatrixXcd(2 * N, 2 * N);
    // add block matrices
    D.block(0, 0, N, N) = Eigen::MatrixXcd::Zero(N, N);
    D.block(0, N, N, N) = I * X + muId;
    D.block(N, 0, N, N) = I * Y + muId;
    D.block(N, N, N, N) = Eigen::MatrixXcd::Zero(N, N);

    // return Dirac operator
    return D;
}

// -----------------------------------------------------------------------------------------------------------------------------------

// fermion determinant
auto FermionDeterminant(Eigen::MatrixXd const &massSqId, Eigen::MatrixXd const muId, Eigen::MatrixXcd const &X, Eigen::MatrixXcd const &Y)
{
    // return the fermion determinant through a simplified expression for the determinant
    return (massSqId - (I * Y + muId) * (I * X + muId)).fullPivLu().determinant();
}

// -----------------------------------------------------------------------------------------------------------------------------------

// chiral condensate
auto ChiralCondensate(Eigen::MatrixXcd const &MInverse)
{
    return MInverse.trace();
}

// number density
auto NumberDensity(int const &N, Eigen::MatrixXcd const &MInverse)
{
    return (MInverse.block(0, N, N, N).trace() + MInverse.block(N, 0, N, N).trace());
}

// -----------------------------------------------------------------------------------------------------------------------------------

// computing the gradient in the parameter space for AdaDelta optimisation
auto Gradient(int const &N, int const &Nf, Eigen::MatrixXcd const &X, Eigen::MatrixXcd const &Y, Eigen::MatrixXcd const &fermionMatInvPos, Eigen::MatrixXcd const &fermionMatInvNeg)
{
    // precomputing quantities used more than once
    std::complex<double> TrX = X.trace();
    std::complex<double> TrY = Y.trace();
    Eigen::MatrixXcd matSum = fermionMatInvPos + fermionMatInvNeg;

    // gradient vector
    Eigen::VectorXcd gradVec(2);

    // computing components
    gradVec(0) = -(double)N * (I * (TrX + TrY)).real() - (double)Nf / 2. * (matSum.block(0, N, N, N).trace() + matSum.block(N, 0, N, N).trace());
    gradVec(1) = -(double)N * (TrX - TrY).real() + (double)Nf / 2. * I * (matSum.block(0, N, N, N).trace() - matSum.block(N, 0, N, N).trace());
    // gradVec *= -1.;

    // return gradient vector
    return gradVec;
}

// decaying average
auto DecayingAverage(double const &g, double const &prev, double const &next)
{
    return g * prev + (1 - g) * next;
}

// -----------------------------------------------------------------------------------------------------------------------------------

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
    // number of configurations to discard in order to decrease autocorrelation
    int tau = std::atoi(argv[5]);
    // number of flavours
    int Nf = std::atoi(argv[6]);
    // number of optimisation steps
    int NOpt = std::atoi(argv[7]);
    // initial countour deformation parameters
    double k1 = std::atof(argv[8]);
    double k2 = std::atof(argv[9]);
    // paramter vector
    Eigen::VectorXd paramVec(2);
    paramVec << k1, k2;
    // change in parameters after a single update
    Eigen::VectorXd deltaParam = Eigen::VectorXd::Zero(2);
    // mean gradient vectors
    Eigen::VectorXd meanGradVec = Eigen::VectorXd::Zero(2);
    // identity matrix
    Eigen::MatrixXcd id = Eigen::MatrixXcd::Identity(N, N);

    // mass, squared mass and chemical potential matrices proportional to the identity
    Eigen::MatrixXd massId = mass * Eigen::MatrixXd::Identity(N, N);
    Eigen::MatrixXd massSqId = sq(mass) * Eigen::MatrixXd::Identity(N, N);
    Eigen::MatrixXd muId = mu * Eigen::MatrixXd::Identity(N, N);

    // -----------------------------------------------------------------------------------------------------------------------------------

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

    // -----------------------------------------------------------------------------------------------------------------------------------

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

    // -----------------------------------------------------------------------------------------------------------------------------------

    // decaying averages
    Eigen::VectorXd dAvgParamSq_prev = Eigen::VectorXd::Zero(2);
    Eigen::VectorXd dAvgParamSq_next = Eigen::VectorXd::Zero(2);
    Eigen::VectorXd dAvgGradSq_prev = Eigen::VectorXd::Zero(2);
    Eigen::VectorXd dAvgGradSq_next = Eigen::VectorXd::Zero(2);
    // run optimisation
    for (int iOpt = 0; iOpt < NOpt; iOpt++)
    {

        // simulations starts here
        //
        // chiral condensate container for reweighting to the original theory from the phase quenched theory
        // std::vector<std::complex<double>> sigma(0);
        // number density container for reweighting to the original theory from the phase quenched theory
        // std::vector<std::complex<double>> density(0);
        // fermion determinant container for reweighting to the original theory from the phase quenched theory
        // std::vector<std::complex<double>> fermionDet(0);
        // gradient vector container for AdaDelta optimisation
        std::vector<Eigen::VectorXcd> gradVecs(0);
        // Dirac operator eigenvalue container (no reweighting ~ not expectation value)
        // std::vector<std::complex<double>> DiracEigen(0);

        // preallocating memory for the used matrices and initialising ~ avoiding repeated memory allocation
        // random matrices
        Eigen::MatrixXcd alpha = RandMatrix() + I * paramVec(0) * id;
        Eigen::MatrixXcd beta = RandMatrix() + I * paramVec(1) * id;
        // complexified matrices
        Eigen::MatrixXcd X = alpha + I * beta;
        Eigen::MatrixXcd Y = alpha.transpose() - I * beta.transpose();
        // random matrices for the next sweep
        Eigen::MatrixXcd alphaNew(N, N);
        Eigen::MatrixXcd betaNew(N, N);
        // complexified matrices for the next sweep
        Eigen::MatrixXcd XNew(N, N);
        Eigen::MatrixXcd YNew(N, N);
        // fermionic operator matrix with +mu ~ M = D + m
        Eigen::MatrixXcd MPos(2 * N, 2 * N);
        // fermionic operator matrix with -mu ~ M = D + m
        Eigen::MatrixXcd MNeg(2 * N, 2 * N);
        // Dirac operator matrix ~ D
        // Eigen::MatrixXcd D(2 * N, 2 * N);
        // inverse of fermionic matrices
        Eigen::MatrixXcd MPosInverse(2 * N, 2 * N);
        Eigen::MatrixXcd MNegInverse(2 * N, 2 * N);
        // fermion determinant
        std::complex<double> det = FermionDeterminant(massSqId, muId, X, Y);
        // bosonic action
        std::complex<double> bosonic = ActionBosonic(N, alpha, beta);
        // fermionic action for the simulated theory
        double fermionic = ActionFermionic(Nf, std::abs(det));

        // sweeps & measurements
        for (int t = 0; t < T; t++)
        {
            // Metropolis step
            // new matrix, adjoint matrix, fermion determinant and action
            alphaNew = alpha + eta * RandMatrix();
            betaNew = beta + eta * RandMatrix();
            XNew = alphaNew + I * betaNew;
            YNew = alphaNew.transpose() - I * betaNew.transpose();
            std::complex<double> detNew = FermionDeterminant(massSqId, muId, XNew, YNew);
            std::complex<double> bosonicNew = ActionBosonic(N, alphaNew, betaNew);
            double fermionicNew = ActionFermionic(Nf, std::abs(detNew));

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
                X = XNew;
                Y = YNew;
                det = detNew;
                bosonic = bosonicNew;
                fermionic = fermionicNew;
            }

            // measurements
            //
            if ((t % tau) == 0)
            {
                // reweighting factor
                // std::complex<double> reweightingFactor = std::pow(det / std::abs(det), Nf) * std::exp(-I * bosonic.imag());
                // fermionic operators
                MPos = FermionicOperator(N, massId, muId, X, Y);
                MNeg = FermionicOperator(N, massId, -muId, X, Y);
                // inverse of fermionic operator
                MPosInverse = MPos.fullPivLu().inverse();
                MNegInverse = MNeg.fullPivLu().inverse();

                // prefactor
                // double pf = Nf / 2. / N;
                // measuring chiral condensate for reweighting
                // sigma.push_back(pf * ChiralCondensate(MPosInverse) * reweightingFactor);
                // measuring number density for reweighting
                // density.push_back((pf * NumberDensity(N, MPosInverse) + mu) * reweightingFactor);
                // measuring reweighting factor
                // fermionDet.push_back(reweightingFactor);
                // measuring gradient vectors
                gradVecs.push_back(Gradient(N, Nf, X, Y, MPosInverse, MNegInverse));

                // Dirac operator & eigenvalues
                /*
                D = DiracOperator(N, muId, alpha + I * beta, alpha.transpose() - I * beta.transpose());
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

        // computing mean of measured gradient vectors
        meanGradVec = std::accumulate(gradVecs.begin(), gradVecs.end(), Eigen::Vector2cd::Zero().eval()).real() / static_cast<double>(gradVecs.size());
        // Eigen::VectorXd meanGradVecImag = std::accumulate(gradVecs.begin(), gradVecs.end(), Eigen::Vector2cd::Zero().eval()).imag() / static_cast<double>(gradVecs.size());

        // std::cout << meanGradVec(0) << " " << meanGradVecImag(0) << " " << meanGradVec(1) << " " << meanGradVecImag(1) << std::endl;
        std::cout << meanGradVec(0) << " " << meanGradVec(1) << std::endl;

        // double meanChiral = std::accumulate(sigma.begin(), sigma.end(), 0. + I * 0.).real() / static_cast<double>(sigma.size());
        // double meanPhase = std::accumulate(fermionDet.begin(), fermionDet.end(), 0. + I * 0.).real() / static_cast<double>(fermionDet.size());

        // std::cout << meanChiral / meanPhase << std::endl;

        // calculate update in parameters
        for (int iPar = 0; iPar < static_cast<int>(paramVec.size()); iPar++)
        {
            // compute decaying average of the squared gradient
            dAvgGradSq_next(iPar) = DecayingAverage(g, dAvgGradSq_prev(iPar), sq(meanGradVec(iPar)));
            // change in parameters
            deltaParam(iPar) = -std::sqrt((dAvgParamSq_next(iPar) + eps) / (dAvgGradSq_next(iPar) + eps)) * meanGradVec(iPar);
            paramVec(iPar) += deltaParam(iPar);
            // compute decaying average of the squared change of parameters
            dAvgParamSq_next(iPar) = DecayingAverage(g, dAvgParamSq_prev(iPar), sq(deltaParam(iPar)));
        }

        // write parameters to screen
        // std::cout << paramVec(0) << " " << paramVec(1) << std::endl;

        // update parameters
        // paramVec += deltaParam;

        // break if change is small enough
        // if (std::abs(deltaParamNext(0)) < eps && std::abs(deltaParamNext(1)) < eps)
        //    break;

        // updating previous parameter vectors
        dAvgParamSq_prev = dAvgParamSq_next;
        dAvgGradSq_prev = dAvgGradSq_next;

        // writing results to screen
        /*
        // measurements
        for (int i = 0; i < static_cast<int>(fermionDet.size()); i++)
        {
            std::cout << sigma[i].real() << " " << sigma[i].imag() << " "
                      << density[i].real() << " " << density[i].imag() << " "
                      << fermionDet[i].real() << " " << fermionDet[i].imag() << std::endl;
            // std::cout << fermionDet[i].real() << " " << fermionDet[i].imag() << std::endl;
        }
        */
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    // Dirac eigeinvalues
    /*
    for (int i = 0; i < static_cast<int>(DiracEigen.size()); i++)
    {
        std::cout << DiracEigen[i].real() << " " << DiracEigen[i].imag() << std::endl;
    }
    */
}
