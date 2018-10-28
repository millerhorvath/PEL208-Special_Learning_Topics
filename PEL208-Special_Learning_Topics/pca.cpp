#include "pca.h"
#include <vector>

using namespace Eigen;
using namespace std;

bool compEigen(const tuple<int, double *> &A, const tuple<int, double *> &B) {
	return *get<1>(A) > *get<1>(B);
}

mhorvath::PCA::PCA()
{
}

mhorvath::PCA::PCA(const MatrixXd &A)
	: DataAdjust(A.rows(), A.cols()), EigenValues(A.cols()),
	ExplainedVariance(A.cols()), EigenVectors(A.cols(), A.cols()),
	OriginalMean(A.cols()), Covariance(A.cols(), A.cols())
{
	MatrixXd evecs(A.cols(), A.cols()); // Unordered eigenvectors
	VectorXd evals(A.cols()); // Unordered eigenvalues
	vector<tuple<int, double *>> ord; // Vector to support eigen values/vectors sorting
	double evalues_sum(0.0); // Sum of eigenvalues (Used to compute proportional )

	// Compute the mean value of each column
	this->OriginalMean = A.colwise().mean();

	// Subtract the mean
	this->DataAdjust = A.rowwise() - this->OriginalMean;

	// Compute covariance matrix
	for (int i = 0; i < A.cols(); i++) {
		for (int j = 0; j < A.cols(); j++) {
			Covariance(i, j) = (this->DataAdjust.col(i).array() *
				this->DataAdjust.col(j).array()).sum() / (A.rows() - 1);
		}
	}

	// Compute eigenvalues and eigenvectors
	EigenSolver<MatrixXd> es(Covariance);
	evals = es.eigenvalues().real();
	evecs = es.eigenvectors().real();

	evalues_sum = evals.sum();

	// Sort eigenvalues and eigenvectors
	for (int i = 0; i < A.cols(); i++) {
		ord.push_back(tuple<int, double *>(i, &evals(i)));
	}

	std::sort(ord.begin(), ord.end(), compEigen);

	for (int i = 0; i < A.cols(); i++) {
		int ind = get<0>(ord[i]);
		this->EigenVectors.col(i) = evecs.col(ind);
		this->EigenValues(i) = evals(ind);
		this->ExplainedVariance(i) = evals(ind) / evalues_sum;
	}
}

mhorvath::PCA::~PCA()
{
}

mhorvath::PCA::PCA(const PCA &P)
	:DataAdjust(P.DataAdjust), EigenVectors(P.EigenVectors), Covariance(P.Covariance),
	EigenValues(P.Covariance), ExplainedVariance(P.Covariance), OriginalMean(P.Covariance) { }

mhorvath::PCA mhorvath::PCA::operator=(const PCA &P)
{
	this->DataAdjust = P.DataAdjust;
	this->EigenVectors = P.EigenVectors;
	this->Covariance = P.Covariance;
	this->EigenValues = P.Covariance;
	this->ExplainedVariance = P.Covariance;
	this->OriginalMean = P.Covariance;

	return *this;
}


MatrixXd mhorvath::PCA::components()
{
	return this->EigenVectors;
}

MatrixXd mhorvath::PCA::covariance()
{
	return this->Covariance;
}
VectorXd mhorvath::PCA::values()
{
	return this->EigenValues;
}

VectorXd mhorvath::PCA::explained_variace_ratio()
{
	return this->ExplainedVariance;
}

MatrixXd mhorvath::PCA::getDataAdjust()
{
	return this->DataAdjust;
}

RowVectorXd mhorvath::PCA::getOriginalMean()
{
	return this->OriginalMean;
}

MatrixXd mhorvath::PCA::transform(const int & n_comp)
{
	return (this->EigenVectors.block(0, 0, EigenVectors.rows(), n_comp).transpose() *
		this->DataAdjust.transpose()).transpose();

	//return this->DataAdjust * this->EigenVectors.block(0, 0, EigenVectors.rows(), n_comp);
}

MatrixXd mhorvath::PCA::rebuild(const MatrixXd &A)
{
	//return (this->EigenVectors.inverse().block(0, 0, this->EigenVectors.rows(), A.cols()) *
	//	A.transpose()).transpose().rowwise() + this->OriginalMean;

	return (this->EigenVectors.transpose().inverse().block(0, 0, this->EigenVectors.rows(), A.cols()) *
		A.transpose()).transpose().rowwise() + this->OriginalMean;
}
