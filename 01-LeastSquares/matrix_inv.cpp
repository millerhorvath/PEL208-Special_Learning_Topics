#include "matrix_inv.h"

using namespace Eigen;

double mhorvath::min_positive(const Eigen::VectorXd &v)
{
	double min(v(0));

	for (int i = 1; i < v.size(); i++) {
		if (v(i) >= 0) {
			min = std::min(min, v(i));
		}
	}

	return min;
}

MatrixXd mhorvath::pinv(const MatrixXd &A) {
	JacobiSVD<MatrixXd> svd(A, ComputeThinU | ComputeThinV); // Apply Singular Value Decomposition in G
	double tol(DBL_EPSILON * (double)std::max(A.rows(), A.cols())*svd.singularValues().maxCoeff()); // Non-zero tolerance
	MatrixXd Sigma(ArrayXXd::Zero(A.rows(), A.cols())); // Sigma matrix fom SVD (svd(A) = U * Sigma * V')

	// Building Sigma pseudo-inverse matrix by adding singular values reciprocals in the main diagonal
	for (int i = 0; i < svd.singularValues().size(); i++) {
		// If singular value i is not 0 (zero)
		if (fabs(svd.singularValues()(i)) >= tol) {
			Sigma(i, i) = 1.0 / svd.singularValues()(i);
		}
		else {
			Sigma(i, i) = 0.0;
		}
	}

	// Return the pseudo-inverse of matrix A (A+ = V * Sigma+ * U')
	return svd.matrixV() * Sigma.transpose() * svd.matrixU().transpose();
}

bool compare(const VectorXd &A, const VectorXd &B) {
	return (A(0) != 0.0 && fabs(A(0)) < fabs(B(0)));
}

MatrixXd mhorvath::inv(const Eigen::MatrixXd &A)
{
	if (A.rows() != A.cols()) {
		throw "Cannot compute the inverse of a non-square matrix";
	}

	int n((int)A.rows());
	MatrixXd AI(n, n * 2);
	std::vector<VectorXd> sorted;

	// Concatenate matrix A and I into AI
	AI << A, MatrixXd::Identity(n, n);

	// Pivoting
	for (int i = 0; i < n; i++) {
		sorted.push_back(AI.row(i));
	}

	sort(sorted.begin(), sorted.end(), compare);

	for (int i = 0; i < n; i++)
	{
		AI.row(i) = sorted[i];
	}

	// Reducing to diagnal matrix
	for (int i = 0; i < n; i++)
	{
		for (int j = n-1; j >= 0; j--) {
			if (j != i && AI(j, i) != 0.0)
			{
				if (AI(i, i) == 0.0) {
					throw "Singular matrix can not be inverted";
				}

				double d = AI(j, i) / AI(i, i);
				AI.row(j) -= d * AI.row(i);
			}
		}
	}

	// Reducing to unit matrix
	for (int i = 0; i < n; i++)
	{
		double d = AI(i, i);

		if (d != 1.0) {
			AI.row(i) /= d;
		}
	}

	// Mathlab equivalent: AI(:, n+1:n*2)
	return MatrixXd(AI.block(0, n, n, n));
}
