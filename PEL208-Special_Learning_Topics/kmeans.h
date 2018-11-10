#pragma once
#include<vector>
#include<Eigen/Dense>

namespace mhorvath {
	class KMeans {
	private:
		std::vector<Eigen::RowVectorXd> centroids;
		unsigned int k;
		KMeans();
	public:
		KMeans(const Eigen::MatrixXd &, const unsigned int &, const unsigned int &max_i = 100);

		unsigned int classify(const Eigen::RowVectorXd &);
	};
}