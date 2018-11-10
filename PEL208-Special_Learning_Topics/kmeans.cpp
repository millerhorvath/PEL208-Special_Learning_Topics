#include "kmeans.h"
#include<ctime>
#include<set>
#include<iostream>

using namespace std;
using namespace Eigen;

typedef vector<RowVectorXd>::iterator c_it;

double euclideanDist(const RowVectorXd &A, const RowVectorXd &B) {
	return (A - B) * (A - B).transpose();
}

double manhattanDist(const RowVectorXd &A, const RowVectorXd &B) {
	return (A - B).cwiseAbs().sum();
}

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
	: k(k)
{
	srand(static_cast<unsigned int>(time(0)));
	this->centroids.resize(k);
	set<unsigned int> temp;

	// Loop to randomly pick k points of X as initial centroids
	for (unsigned int i = 0; i < k; i++) {
		unsigned int idx = rand() % X.rows();

		// Guaranteeing to pick three different indices of X to be used as initial centroids
		if (temp.find(idx) == temp.end()) {
			temp.insert(idx);
			centroids[i] = X.row(idx);

			cout << idx << endl;
		}
		else {
			i--;
		}
	}
	cout << endl;
	temp.clear();

	vector<RowVectorXd> old_centroids; // Used to check if the centroids changed after the iteration
	vector<unsigned int> clusters(X.rows()); // Stores the observation cluster at each iteration

	unsigned int it_count = 0;

	// Update centroids until they converge
	do {
		old_centroids.clear();
		old_centroids.assign(this->centroids.begin(), this->centroids.end());
		vector<unsigned int> clusters_size(k, 0); // Stores the number of observation in each cluster

		// Classify observations using the actual centroids
		for (unsigned int i = 0; i < X.rows(); i++) {
			clusters[i] = this->classifyVector(X.row(i));
			clusters_size[clusters[i]]++;
		}

		// Reset centroids
		for (unsigned int i = 0; i < k; i++) {
			this->centroids[i] = RowVectorXd::Zero(X.cols());
		}

		// Update Centroids based on the clusters
		for (unsigned int i = 0; i < X.rows(); i++) {
			centroids[clusters[i]] += X.row(i);
		}

		for (unsigned int i = 0; i < k; i++) {
			this->centroids[i] /= clusters_size[i];
		}

		//// Print centroid update
		//for (unsigned int i = 0; i < k; i++) {
		//	cout << old_centroids[i] << endl << this->centroids[i] << endl;
		//}
		//cout << endl;

		//system("pause");
		
		it_count++;
	} while (!(compCentroids(centroids, old_centroids) || it_count == max_i));

	if (it_count == max_i) {
		cout << "KMeans did not converge after " << max_i << " iterations!" << endl << endl;
	}
}

unsigned int mhorvath::KMeans::classifyVector(const RowVectorXd &X)
{
	//double closest_dist = euclideanDist(X, this->centroids[0]);
	double closest_dist = manhattanDist(X, this->centroids[0]);
	unsigned int closest_id = 0;

	for (unsigned int i = 1; i < this->k; i++) {
		//const double temp_dist(euclideanDist(X, this->centroids[i]));
		const double temp_dist(manhattanDist(X, this->centroids[i]));

		if (temp_dist < closest_dist) {
			closest_dist = temp_dist;
			closest_id = i;
		}
	}

	return closest_id;
}

vector<unsigned int> mhorvath::KMeans::classifyMatrix(const Eigen::MatrixXd &X)
{
	vector<unsigned int> classes(X.rows());

	for (unsigned int i = 0; i < X.rows(); i++) {
		classes[i] = this->classifyVector(X.row(i));
	}
	
	return classes;
}

std::vector<Eigen::RowVectorXd> mhorvath::KMeans::getCentroids()
{
	return this->centroids;
}
