#pragma once
#include <Eigen/Dense>
#include "matrix_inv.h"

namespace mhorvath {
	// Linear least squares (can be used to find a linear approximation of polynomials)
	Eigen::VectorXd leastSquares(const Eigen::MatrixXd &, const Eigen::VectorXd &);

	// Linear least squares (can be used to find a linear approximation of polynomials)
	Eigen::VectorXd leastSquares_pinv(const Eigen::MatrixXd &, const Eigen::VectorXd &);

	// Weighted linear regression
	Eigen::VectorXd weightedLeastSquares(const Eigen::MatrixXd &, const Eigen::VectorXd &, const Eigen::VectorXd &);

	// Weighted linear regression
	Eigen::VectorXd weightedLeastSquares_pinv(const Eigen::MatrixXd &, const Eigen::VectorXd &, const Eigen::VectorXd &);

	// Weighted linear regression with default weights (1 / (y - X*B))
	Eigen::VectorXd weightedLeastSquares(const Eigen::MatrixXd &, const Eigen::VectorXd &);

	// Weighted linear regression with default weights (1 / (y - X*B))
	Eigen::VectorXd weightedLeastSquares_pinv(const Eigen::MatrixXd &, const Eigen::VectorXd &);
}
