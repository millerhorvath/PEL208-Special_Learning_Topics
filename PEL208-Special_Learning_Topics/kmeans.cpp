#include "kmeans.h"
#include<ctime>
#include<set>
#include<iostream>

using namespace std;
using namespace Eigen;

typedef vector<RowVectorXd>::iterator c_it;

bool compCentroids(const vector<RowVectorXd> &A, const vector<RowVectorXd> &B) {
	bool ret = true;

	for (unsigned int i = 0; ret && i < A.size(); i++) {
		if (!A[i].isApprox(B[i])) {
			ret = false;
		}
	}

	return ret;
}

mhorvath::KMeans::KMeans(const MatrixXd &X, const unsigned int &k, const unsigned int &max_i)
{
	srand(static_cast<unsigned int>(time(0)));
	this->centroids.resize(k);
	set<unsigned int> temp;
	set<unsigned int>::iterator it_temp;

	// Loop to randomly pick k points of X as initial centroids
	for (unsigned int i = 0; i < k; i++) {
		unsigned int idx = rand() % X.rows();

		// Guaranteeing to pick three different indices of X to be used as initial centroids
		if (temp.find(idx) == temp.end()) {
			temp.insert(idx);
			centroids[i] = X.row(idx);
			cout << idx << " ";
		}
		else {
			i--;
		}
	}

	cout << endl;

	for (c_it it = this->centroids.begin(); it != this->centroids.end(); it++) {
		cout << *it << endl << endl;
	}

	//do {
	//	vector<RowVectorXd> old_centroids(centroids);
	//} while (!compCentroids(centroids, old_centroids));
}
