#pragma once
#include <Eigen/Dense>

namespace mhorvath {
	class PCA {
	private:
		Eigen::MatrixXd DataAdjust; // Data - mean
		Eigen::MatrixXd EigenVectors;
		Eigen::MatrixXd Covariance;
		Eigen::VectorXd EigenValues;
		Eigen::VectorXd ExplainedVariance; // Normalized EigenValues
		Eigen::RowVectorXd OriginalMean; // Mean of the original data features

		// Default constructor not allowed
		PCA();

	public:
		~PCA();
		PCA(const PCA &);
		PCA operator=(const PCA &);

		PCA(const Eigen::MatrixXd &);

		Eigen::MatrixXd getDataAdjust();
		Eigen::MatrixXd components(); // Returns the eigenvectors
		Eigen::MatrixXd covariance(); // Returns the covariance matrix
		Eigen::VectorXd values(); // Returns eigenvalues
		Eigen::VectorXd explained_variace_ratio(); // Returns the explained variance os each component
		Eigen::RowVectorXd getOriginalMean();

		Eigen::MatrixXd transform(const int &);
		Eigen::MatrixXd rebuild(const Eigen::MatrixXd &);
	};
}
