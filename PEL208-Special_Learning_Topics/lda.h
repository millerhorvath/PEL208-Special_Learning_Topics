#pragma once
#include "pca.h"
#include <Eigen/Dense>
#include <vector>

namespace mhorvath {
	mhorvath::PCA lda(const Eigen::MatrixXd &, const std::vector<std::string> &);
}