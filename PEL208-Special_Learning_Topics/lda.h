#pragma once
#include <Eigen/Dense>
#include <vector>

namespace mhorvath {
	class LDA {
	private:
		Eigen::MatrixXd DataAdjust;
		Eigen::MatrixXd EigenVectors;
		Eigen::VectorXd EigenValues;
		Eigen::VectorXd ExplainedVariance; // Normalized EigenValues
		Eigen::RowVectorXd DataMean;

		Eigen::MatrixXd Sb;
		Eigen::MatrixXd Sw;

		std::vector<Eigen::RowVectorXd> ClassMean; // Used to compute the mean of the variables in each class

		LDA();
	public:
		LDA(const Eigen::MatrixXd &, const std::vector<std::string> &);

		Eigen::MatrixXd components(); // Returns the eigenvectors
		Eigen::VectorXd values(); // Returns eigenvalues
		Eigen::VectorXd explained_variace_ratio(); // Returns the explained variance os each component
		Eigen::VectorXd data_mean();
		Eigen::MatrixXd classes_mean();

		Eigen::MatrixXd transform(const unsigned int &);
		Eigen::MatrixXd rebuild(const Eigen::MatrixXd &);

		Eigen::MatrixXd getSb();
		Eigen::MatrixXd getSw();
	};

	//mhorvath::PCA lda(const Eigen::MatrixXd &, const std::vector<std::string> &);
}