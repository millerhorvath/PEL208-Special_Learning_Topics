#include "lda.h"
#include <map>
#include <vector>
#include <iostream>
#include "matrix_inv.h"

using namespace Eigen;
using namespace std;

//typedef pair<vector<double>, int> mypair;

bool compEigen2(const tuple<int, double *> &A, const tuple<int, double *> &B) {
	return *get<1>(A) > *get<1>(B);
}

mhorvath::LDA::LDA(const MatrixXd &M, const vector<string> &class_vec)
	: DataAdjust(M.rows(), M.cols()), EigenVectors(M.cols(), M.cols()),
	EigenValues(M.cols()), ExplainedVariance(M.cols()), DataMean(M.colwise().mean()),
	Sb(MatrixXd::Zero(M.cols(), M.cols())), Sw(MatrixXd::Zero(M.cols(), M.cols()))
{
	unsigned const int m((unsigned int)M.rows());
	unsigned const int n((unsigned int)M.cols());
	
	map<string, int> classes; // Map data structure used to check the number of classes
	map<string, int>::iterator it_class; // Iterator for classes map
	map<string, unsigned int> ids; // Map data structure used to split data by class
	map<string, unsigned int>::iterator it_ids; // Iterator for ids map
	map<string, MatrixXd> X; // Map used to store data splited by class
	map<string, MatrixXd>::iterator it_X; // Iterator for X map
	vector<tuple<int, double *>> ord; // Vector to support eigen values/vectors sorting
	
	MatrixXd evecs(n, n); // Unordered eigenvectors
	VectorXd evals(n); // Unordered eigenvalues
	double evalues_sum(0.0); // Sum of eigenvalues (Used to compute proportional explained variance)

	// Subtract the mean
	this->DataAdjust = M.rowwise() - this->DataMean;

	// Check number of classes
	for (unsigned int i = 0; i < m; i++) {
		classes.insert(pair<string, int>(class_vec[i], 0));
	}

	unsigned const int g((unsigned int)classes.size()); // Stores number of classes

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
			it_X->second(it_ids->second, j) = DataAdjust(i, j);
		}
		it_ids->second++;
	}

	this->ClassMean.resize(g);

	// Compute the mean of each class
	it_X = X.begin();
	for (unsigned int i = 0; it_X != X.end(); it_X++, i++) {
		this->ClassMean[i] = VectorXd(it_X->second.colwise().mean());
		//data_mean += class_mean[i];
	}
	//data_mean /= g;

	// Compute Sw matrix
	it_X = X.begin();
	for (unsigned int i = 0; it_X != X.end(); it_X++, i++) {
		MatrixXd temp(it_X->second.transpose().colwise() - this->ClassMean[i].transpose());

		Sw += temp * temp.transpose();
	}

	// Compute Sb matrix
	it_class = classes.begin();
	for (unsigned int i = 0; i < g; i++, it_class++) {
		//RowVectorXd temp = this->ClassMean[i] - this->DataMean;
		RowVectorXd temp = this->ClassMean[i];
		Sb += MatrixXd((temp.transpose() * temp).array() * it_class->second);
	}

	// Compute eigenvalues and eigenvectors
	EigenSolver<MatrixXd> es(Sw.inverse() * Sb);
	evals = es.eigenvalues().real();
	evecs = es.eigenvectors().real();

	evalues_sum = evals.sum();

	// Sort eigenvalues and eigenvectors
	for (int i = 0; i < M.cols(); i++) {
		ord.push_back(tuple<int, double *>(i, &evals(i)));
	}

	std::sort(ord.begin(), ord.end(), compEigen2);

	for (int i = 0; i < M.cols(); i++) {
		int ind = get<0>(ord[i]);
		this->EigenVectors.col(i) = evecs.col(ind);
		this->EigenValues(i) = evals(ind);
		this->ExplainedVariance(i) = evals(ind) / evalues_sum;
	}
}

mhorvath::LDA::LDA() {}

MatrixXd mhorvath::LDA::components()
{
	return this->EigenVectors;
}

VectorXd mhorvath::LDA::values()
{
	return this->EigenValues;
}

VectorXd mhorvath::LDA::explained_variace_ratio()
{
	return this->ExplainedVariance;
}

Eigen::VectorXd mhorvath::LDA::data_mean()
{
	return this->DataMean;
}

Eigen::MatrixXd mhorvath::LDA::classes_mean()
{
	MatrixXd means(this->ClassMean.size(), this->DataAdjust.cols());
	vector<RowVectorXd>::iterator it(this->ClassMean.begin());

	for (int i = 0; it != this->ClassMean.end(); i++, it++) {
		means.row(i) = *it;
	}

	return means;
}

MatrixXd mhorvath::LDA::transform(const unsigned int & n_comp)
{
	return (this->EigenVectors.block(0, 0, EigenVectors.rows(), n_comp).transpose() *
		this->DataAdjust.transpose()).transpose();
}

MatrixXd mhorvath::LDA::rebuild(const MatrixXd &A)
{
	//MatrixXd temp(MatrixXd::Zero(this->EigenVectors.rows(), this->EigenVectors.cols()).array() + 1.0);
	//temp.block(0, 0, temp.rows(), A.cols()) = this->EigenVectors.block(0, 0, temp.rows(), A.cols());

	//return (temp.transpose().inverse().block(0, 0, temp.rows(), A.cols()) *
	//	A.transpose()).transpose();

	//return (this->EigenVectors.transpose().inverse().block(0, 0, this->EigenVectors.rows(), A.cols()) *
	//	A.transpose()).transpose();

	return (this->EigenVectors.transpose().inverse().block(0, 0, this->EigenVectors.rows(), A.cols()) *
		A.transpose()).transpose().rowwise() + this->DataMean;
}

MatrixXd mhorvath::LDA::getSb()
{
	return this->Sb;
}

MatrixXd mhorvath::LDA::getSw()
{
	return this->Sw;
}
