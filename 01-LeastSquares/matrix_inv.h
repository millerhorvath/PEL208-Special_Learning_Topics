#pragma once
#include <Eigen/Dense>
#include <vector>

namespace mhorvath {
	double min_positive(const Eigen::VectorXd &);

	// https://www.youtube.com/watch?v=jYw0OIeRnQE
	// Computes matrix pseudo-inverse with Moore–Penrose inverse through Singular Value Decomposition
	Eigen::MatrixXd pinv(const Eigen::MatrixXd &);

	// Based on: https://www.sanfoundry.com/cpp-program-implement-gauss-jordan-elimination/
	// Based on: https://www.youtube.com/watch?v=bt52Y_1crAo
	// Computes matrix inverse through Gauss-Jordan Elimination 
	Eigen::MatrixXd inv(const Eigen::MatrixXd &);
}