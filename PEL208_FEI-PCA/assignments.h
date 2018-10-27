#pragma once
#define _CRT_SECURE_NO_DEPRECATE
#include <Eigen/Dense>

namespace mhorvath {
	void runPCAExperimentEx(const Eigen::MatrixXd &, const char * const);
	void runPCAExperiment(const Eigen::MatrixXd &, const char * const);

	void runLeastSquaresExperiment(const Eigen::MatrixXd &, const Eigen::VectorXd &, const char * const);

	void inClassExample();

	void alpsWater();

	void booksXgrades();

	void usCensus();

	void hald();
}