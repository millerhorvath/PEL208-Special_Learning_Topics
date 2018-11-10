#pragma once
#define _CRT_SECURE_NO_DEPRECATE
#include <Eigen/Dense>
#include <vector>

namespace mhorvath {
	void runPCAExperimentEx(const Eigen::MatrixXd &, const char * const, const char * const);
	void runPCAExperiment(const Eigen::MatrixXd &, const char * const, const char * const);
	void runLeastSquaresExperiment(const Eigen::MatrixXd &, const Eigen::VectorXd &, const char * const, const char * const);
	void runLDAExperiment(const Eigen::MatrixXd &, const std::vector<std::string> &, const char * const, const char * const);
	void runPCA_LDAExperiment(const Eigen::MatrixXd &, const std::vector<std::string> &, const char * const, const char * const);
	void runKMeans_Experiment(const Eigen::MatrixXd &, const char * const, const char * const);

	void inClassExample();

	void alpsWater();

	void booksXgrades();

	void usCensus();

	void hald();

	void iris();

	void inClassExampleLDA();
}