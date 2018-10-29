#include "lda.h"
#include <map>
#include <vector>
#include <iostream>

using namespace Eigen;
using namespace std;

//typedef pair<vector<double>, int> mypair;

mhorvath::PCA mhorvath::lda(const MatrixXd &M, const vector<string> &class_vec)
{
	unsigned const int m((unsigned int)M.rows());
	unsigned const int n((unsigned int)M.cols());
	map<string, int> classes;
	map<string, int>::iterator it_class;
	map<string, unsigned int> ids;
	map<string, unsigned int>::iterator it_ids;
	map<string, MatrixXd> X;
	map<string, MatrixXd>::iterator it_X;
	vector<VectorXd> class_mean;
	VectorXd data_mean(VectorXd::Zero(n));
	MatrixXd Sb(MatrixXd::Zero(n, n));
	MatrixXd Sw(MatrixXd::Zero(n, n));

	// Check number of classes
	for (unsigned int i = 0; i < m; i++) {
		it_class = classes.find(class_vec[i]);
		if (it_class == classes.end()) {
			classes.insert(pair<string, int>(class_vec[i], 0));
		}
	}

	unsigned const int g((unsigned int) classes.size());
	
	// Compute number of elements of each class
	for (unsigned int i = 0; i < m; i++) {
		it_class = classes.find(class_vec[i]);
		it_class->second++;
	}

	// Split data by class
	for (it_class = classes.begin(); it_class != classes.end(); it_class++) {
		X.insert(pair<string, MatrixXd>(it_class->first, MatrixXd(it_class->second, n)));
		ids.insert(pair<string, unsigned int>(it_class->first, 0));
	}

	for (unsigned int i = 0; i < m; i++) {
		const string c(class_vec[i]);
		it_X = X.find(c);
		it_ids = ids.find(c);
		for (unsigned int j = 0; j < n; j++) {
			it_X->second(it_ids->second, j) = M(i, j);
		}
		it_ids->second++;
	}

	class_mean.resize(g);
	
	it_X = X.begin();
	for(unsigned int i = 0; it_X != X.end(); it_X++, i++) {
		class_mean[i] = VectorXd(it_X->second.colwise().mean());
		data_mean += class_mean[i];
	}
	data_mean /= g;

	for (it_X = X.begin(); it_X != X.end(); it_X++) {
		MatrixXd temp(X.begin()->second.transpose().colwise() - class_mean[0]);

		Sw += temp * temp.transpose();
	}

	//cout << "Sw =" << endl << Sw << endl << endl;

	it_class = classes.begin();
	for (unsigned int i = 0; i < g; i++, it_class++) {
		VectorXd temp = class_mean[i] - data_mean;
		Sb += MatrixXd((temp * temp.transpose()).array() * it_class->second);
	}

	//cout << "Sb =" << endl << Sb << endl << endl;

	return mhorvath::PCA(Sw.inverse() * Sb);
}
