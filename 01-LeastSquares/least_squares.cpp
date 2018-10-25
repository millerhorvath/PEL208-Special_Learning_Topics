#include "least_squares.h"

using namespace Eigen;

VectorXd mhorvath::leastSquares(const MatrixXd &X, const VectorXd &y)
{
	//// General inverse function
	//return VectorXd((X.transpose() * X).inverse() * X.transpose() * y);

	// General inverse function
	return VectorXd(mhorvath::inv(X.transpose() * X) * X.transpose() * y);
}

VectorXd mhorvath::leastSquares_pinv(const MatrixXd &X, const VectorXd &y)
{
	//// Using pseudo-inverse function to avoid singular matrix issues
	//return VectorXd((X.transpose() * X).completeOrthogonalDecomposition().pseudoInverse() *
	//	X.transpose() * y);

	// Using pseudo-inverse function to avoid singular matrix issues
	return VectorXd(mhorvath::pinv(X.transpose() * X) * X.transpose() * y);
}

VectorXd mhorvath::weightedLeastSquares(const MatrixXd &X, const Eigen::VectorXd &y, const Eigen::VectorXd &w) {
	// Convert weight vector into a diagonal matrix
	MatrixXd W = MatrixXd(w.asDiagonal());

	//// General inverse function
	//return VectorXd((X.transpose() * W * X).inverse() * X.transpose() * W * y);

	// General inverse function
	return VectorXd(mhorvath::inv(X.transpose() * W * X) * X.transpose() * W * y);
}

VectorXd mhorvath::weightedLeastSquares_pinv(const MatrixXd &X, const Eigen::VectorXd &y, const Eigen::VectorXd &w) {
	// Convert weight vector into a diagonal matrix
	MatrixXd W = MatrixXd(w.asDiagonal());

	//// Using pseudo-inverse function to avoid singular matrix issues
	//return VectorXd((X.transpose() * W * X).completeOrthogonalDecomposition().pseudoInverse() *
	//	X.transpose() * W * y);

	// Using pseudo-inverse function to avoid singular matrix issues
	return VectorXd(mhorvath::pinv(X.transpose() * W * X) * X.transpose() * W * y);
}

VectorXd mhorvath::weightedLeastSquares(const MatrixXd &X, const Eigen::VectorXd &y) {
	// Find coefficients to compute the default weight vector
	VectorXd B = leastSquares(X, y);

	// Default weight vector
	VectorXd w = 1.0 / (y - X * B).cwiseAbs().array();

	// Compute and return the weighted least squares
	return weightedLeastSquares(X, y, w);
}

VectorXd mhorvath::weightedLeastSquares_pinv(const MatrixXd &X, const Eigen::VectorXd &y) {
	// Find coefficients to compute the default weight vector
	VectorXd B = leastSquares_pinv(X, y);

	// Default weight vector
	VectorXd w = 1.0 / (y - X * B).cwiseAbs().array();

	// Compute and return the weighted least squares
	return weightedLeastSquares_pinv(X, y, w);
}
