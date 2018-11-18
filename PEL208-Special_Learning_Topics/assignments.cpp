#include "assignments.h"
#include "pca.h"
#include "least_squares.h"
#include "lda.h"
#include "kmeans.h"
#include <iostream>
#include <cstdio>
#include <filesystem>

using namespace Eigen;
using namespace std;

void mhorvath::runPCAExperimentEx(const MatrixXd &D, const char * const f_label = "", const char * const out_path = "")
{
	unsigned const int n((unsigned int)D.cols()); // Number of features
	unsigned const int m((unsigned int)D.rows()); // Number of observations
	unsigned const int p_m((unsigned int)std::min((unsigned int)10, m)); // Limit of lines to print
	unsigned const int p_n((unsigned int)std::min((unsigned int)10, n)); // Limit of lines to print
	FILE *f; // File used
	filesystem::path out_file; // Used to build output path
	char f_name[256]; // Used to build output file name

	if (strcmp(out_path, "")) {
		cout << filesystem::create_directory(out_path); // Create output path
	}

	cout << "######### " << f_label << " - PCA EXPERIMENT ########## " << endl << endl;

	mhorvath::PCA pca(D); // Compute PCA

	// Print PCA Data (At most 10 lines and 10 columns)
	cout << "Data =" << endl << D.block(0, 0, p_m, p_n) << endl << endl;
	cout << "Data Mean =" << endl << pca.getOriginalMean().segment(0, p_n) << endl << endl;
	cout << "DataAdjust =" << endl << pca.getDataAdjust().block(0, 0, p_m, p_n) << endl << endl;
	cout << "Covariance =" << endl << pca.covariance().block(0, 0, p_n, p_n) << endl << endl;
	cout << "Eigenvalues =" << endl << pca.values().segment(0, p_n) << endl << endl;
	cout << "Eigenvectors =" << endl << pca.components().block(0, 0, p_n, p_n) << endl << endl;
	cout << "Explained Variance =" << endl << pca.explained_variace_ratio().segment(0, p_n) << endl << endl;
	cout << "Sum of Explained Variance =" << endl << pca.explained_variace_ratio().sum() << endl << endl;

	// Vector of matrices to store dimensionality reducted data using from 1 to n principal components
	MatrixXd *reduction_D = new MatrixXd[D.cols()];

	// Compute dimensionality reduction
	for (int i = 0; i < D.cols(); i++) {
		reduction_D[i] = pca.transform(i + 1);
	}

	// Print reducted data
	for (unsigned int i = n; i > 0; i--) {
		printf("Dimensionality reduction (%d components) =\n", i);
		cout << reduction_D[i - 1].block(0, 0, p_m, i) << endl << endl;
	}

	// Vector of matrices to store rebuilt data using from 1 to n principal components
	MatrixXd *rebuilt_D = new MatrixXd[D.cols()];

	// Compute data rebuilding
	for (unsigned int i = 0; i < n; i++) {
		rebuilt_D[i] = pca.rebuild(reduction_D[i]);
	}

	// Print rebuilt data
	for (unsigned int i = n; i > 0; i--) {
		printf("Rebuild Original Data (%d components) =\n", i);
		cout << rebuilt_D[i - 1].block(0, 0, p_m, p_n) << endl << endl;
	}


	// Write all results in file
	sprintf(f_name, "%s_components.csv", f_label);

	cout << "GENERATED FILES: " << endl;

	out_file = filesystem::current_path() / out_path / f_name;
	f = fopen(out_file.string().c_str(), "w");

	for (int c = 0; c < pca.components().cols(); c++) {
		fprintf(f, "%f", pca.components()(0, c));
		for (int l = 1; l < pca.components().rows(); l++) {
			fprintf(f, ",%f", pca.components()(l, c));
		}
		fprintf(f, "\n");
	}

	fclose(f);

	cout << f_name << endl;


	for (int i = 0; i < D.cols(); i++) {
		sprintf(f_name, "%s_reduction_comp_%d.csv", f_label, i + 1);
		out_file = filesystem::current_path() / out_path / f_name;

		f = fopen(out_file.string().c_str(), "w");

		for (int l = 0; l < reduction_D[i].rows(); l++) {
			fprintf(f, "%f", reduction_D[i](l, 0));
			for (int c = 1; c < reduction_D[i].cols(); c++) {
				fprintf(f, ",%f", reduction_D[i](l, c));
			}
			fprintf(f, "\n");
		}

		fclose(f);
		cout << f_name << endl;
	}
	cout << endl;


	for (int i = 0; i < D.cols(); i++) {
		sprintf(f_name, "%s_rebuilt_comp_%d.csv", f_label, i + 1);
		out_file = filesystem::current_path() / out_path / f_name;

		f = fopen(out_file.string().c_str(), "w");

		for (int l = 0; l < rebuilt_D[i].rows(); l++) {
			fprintf(f, "%f", rebuilt_D[i](l, 0));
			for (int c = 1; c < rebuilt_D[i].cols(); c++) {
				fprintf(f, ",%f", rebuilt_D[i](l, c));
			}
			fprintf(f, "\n");
		}

		fclose(f);
		cout << f_name << endl;
	}
	cout << endl;

	system("PAUSE");
	system("CLS");
}

void mhorvath::runPCAExperiment(const MatrixXd &D, const char * const f_label = "", const char * const out_path = "")
{
	unsigned const int n((unsigned int)D.cols()); // Number of features
	unsigned const int m((unsigned int)D.rows()); // Number of observations
	unsigned const int p_m((unsigned int)std::min((unsigned int)10, m)); // Limit of lines to print
	unsigned const int p_n((unsigned int)std::min((unsigned int)10, n)); // Limit of lines to print
	FILE *f; // File used
	filesystem::path out_file; // Used to build output path
	char f_name[256]; // Used to build output file name

	if (strcmp(out_path, "")) {
		cout << filesystem::create_directory(out_path); // Create output path
	}

	cout << "######### " << f_label << " - PCA EXPERIMENT ########## " << endl << endl;

	mhorvath::PCA pca(D); // Compute PCA

	// Print PCA Data (At most 10 lines and 10 columns)
	cout << "Data =" << endl << D.block(0, 0, p_m, p_n) << endl << endl;
	cout << "Data Mean =" << endl << pca.getOriginalMean().segment(0, p_n) << endl << endl;
	cout << "DataAdjust =" << endl << pca.getDataAdjust().block(0, 0, p_m, p_n) << endl << endl;
	cout << "Covariance =" << endl << pca.covariance().block(0, 0, p_n, p_n) << endl << endl;
	cout << "Eigenvalues =" << endl << pca.values().segment(0, p_n) << endl << endl;
	cout << "Eigenvectors =" << endl << pca.components().block(0, 0, p_n, p_n) << endl << endl;
	cout << "Explained Variance =" << endl << pca.explained_variace_ratio().segment(0, p_n) << endl << endl;
	cout << "Sum of Explained Variance =" << endl << pca.explained_variace_ratio().sum() << endl << endl;

	// Vector of matrices to store dimensionality reducted data using from 1 to n principal components
	MatrixXd *reduction_D = new MatrixXd[D.cols()];

	// Compute dimensionality reduction
	for (int i = 0; i < D.cols(); i++) {
		reduction_D[i] = pca.transform(i + 1);
	}

	// Print reducted data
	for (unsigned int i = n; i > 0; i--) {
		printf("Dimensionality reduction (%d components) =\n", i);
		cout << reduction_D[i - 1].block(0, 0, p_m, i) << endl << endl;
	}

	// Vector of matrices to store rebuilt data using from 1 to n principal components
	MatrixXd *rebuilt_D = new MatrixXd[D.cols()];

	// Compute data rebuilding
	for (unsigned int i = 0; i < n; i++) {
		rebuilt_D[i] = pca.rebuild(reduction_D[i]);
	}

	// Print rebuilt data
	for (unsigned int i = n; i > 0; i--) {
		printf("Rebuild Original Data (%d components) =\n", i);
		cout << rebuilt_D[i - 1].block(0, 0, p_m, p_n) << endl << endl;
	}

	sprintf(f_name, "%s_components.csv", f_label);

	// Write all results in file
	cout << "GENERATED FILES: " << endl;

	out_file = filesystem::current_path() / out_path / f_name;
	f = fopen(out_file.string().c_str(), "w");

	for (int c = 0; c < pca.components().cols(); c++) {
		fprintf(f, "%f", pca.components()(0, c));
		for (int l = 1; l < pca.components().rows(); l++) {
			fprintf(f, ",%f", pca.components()(l, c));
		}
		fprintf(f, "\n");
	}

	fclose(f);
	cout << f_name << endl;

	cout << endl;

	system("PAUSE");
	system("CLS");
}

void mhorvath::runLeastSquaresExperiment(const MatrixXd &X, const VectorXd &y, const char * const f_label = "", const char * const out_path = "")
{
	unsigned const int n((unsigned int)X.cols()); // Number of features
	unsigned const int m((unsigned int)X.rows()); // Number of observations
	unsigned const int p_m((unsigned int)std::min((unsigned int)10, m)); // Limit of lines to print
	unsigned const int p_n((unsigned int)std::min((unsigned int)10, n)); // Limit of lines to print
	VectorXd Beta(n); // Least squares coefficients
	FILE *f; // File used
	filesystem::path out_file; // Used to build output path
	char f_name[256]; // Used to build output file name

	if (strcmp(out_path, "")) {
		cout << filesystem::create_directory(out_path); // Create output path
	}

	cout << "######### " << f_label << " - LEAST SQUARES EXPERIMENT ########## " << endl << endl;

	// Fit data into a line using least squares regression
	Beta = mhorvath::leastSquares(X, y);

	// Print Least Squares Data (At most 10 lines and 10 columns)
	cout << "X =" << endl << X.block(0, 0, p_m, p_n) << endl << endl;
	cout << "y =" << endl << y.segment(0, p_m) << endl << endl;
	cout << "Beta =" << endl << Beta.segment(0, p_n) << endl << endl;

	sprintf(f_name, "%s_coefs.csv", f_label); // Build output file name

	// Write all results in file
	cout << "GENERATED FILES: " << endl;

	out_file = filesystem::current_path() / out_path / f_name;
	f = fopen(out_file.string().c_str(), "w");

	fprintf(f, "%f", Beta(0));
	for (int i = 1; i < Beta.size(); i++) {
		fprintf(f, ",%f", Beta(i));
	}
	fprintf(f, "\n");

	fclose(f);
	cout << f_name << endl << endl;

	system("PAUSE");
	system("CLS");
}

void mhorvath::runLDAExperiment(const MatrixXd &X, const vector<string> &classes, const char * const f_label = "", const char * const out_path = "")
{
	unsigned const int n((unsigned int)X.cols()); // Number of features
	unsigned const int m((unsigned int)X.rows()); // Number of observations
	unsigned const int p_m((unsigned int)std::min((unsigned int)10, m)); // Limit of lines to print
	unsigned const int p_n((unsigned int)std::min((unsigned int)10, n)); // Limit of lines to print
	MatrixXd lda_comp(n, n);
	FILE *f; // File used
	filesystem::path out_file; // Used to build output path
	char f_name[256]; // Used to build output file name

	if (strcmp(out_path, "")) {
		filesystem::create_directory(out_path); // Create output path
	}

	cout << "######### " << f_label << " - LDA EXPERIMENT ########## " << endl << endl;

	mhorvath::LDA lda(X, classes);

	cout << "Data =" << endl << X.block(0, 0, p_m, p_n) << endl << endl;
	cout << "Sb =" << endl << lda.getSb() << endl << endl;
	cout << "Sw =" << endl << lda.getSw() << endl << endl;
	cout << "Data Mean =" << endl << lda.data_mean() << endl << endl;
	cout << "Classes Mean =" << endl << lda.classes_mean() << endl << endl;
	cout << "Eigenvalues =" << endl << lda.values().segment(0, p_n) << endl << endl;
	cout << "Eigenvectors =" << endl << lda.components().block(0, 0, p_n, p_n) << endl << endl;
	cout << "Explained Variance =" << endl << lda.explained_variace_ratio().segment(0, p_n) << endl << endl;
	cout << "Sum of Explained Variance =" << endl << lda.explained_variace_ratio().sum() << endl << endl;

	// Vector of matrices to store dimensionality reducted data using from 1 to n principal components
	MatrixXd *reduction_D = new MatrixXd[X.cols()];

	// Compute dimensionality reduction
	for (int i = 0; i < X.cols(); i++) {
		reduction_D[i] = lda.transform(i + 1);

		//cout << "Reduction " << i+1 << "=" << endl << reduction_D[i] << endl << endl;
	}

	// Vector of matrices to store rebuilt data using from 1 to n principal components
	MatrixXd *rebuilt_D = new MatrixXd[X.cols()];

	// Compute data rebuilding
	for (unsigned int i = 0; i < n; i++) {
		rebuilt_D[i] = lda.rebuild(reduction_D[i]);

		//cout << "Rebuild " << i + 1 << "=" << endl << rebuilt_D[i] << endl << endl;
	}

	// Write all results in file
	sprintf(f_name, "%s_lda_vectors.csv", f_label); // Build output file name

	cout << "GENERATED FILES: " << endl;

	out_file = filesystem::current_path() / out_path / f_name;
	f = fopen(out_file.string().c_str(), "w");

	lda_comp = lda.components();

	for (int j = 0; j < lda_comp.rows(); j++) {
		fprintf(f, "%f", lda_comp(j, 0));
		for (int i = 1; i < lda_comp.cols(); i++) {
			fprintf(f, ",%f", lda_comp(j, i));
		}
		fprintf(f, "\n");
	}

	fclose(f);
	cout << f_name << endl;

	for (unsigned int i = 0; i < n; i++) {
		sprintf(f_name, "%s_reduction_lda_%d.csv", f_label, i + 1);
		out_file = filesystem::current_path() / out_path / f_name;

		f = fopen(out_file.string().c_str(), "w");

		for (int l = 0; l < reduction_D[i].rows(); l++) {
			//fprintf(f, "%f", reduction_D[i](l, 0));
			for (int c = 0; c < reduction_D[i].cols(); c++) {
				fprintf(f, "%f,", reduction_D[i](l, c));
			}
			fprintf(f, "%s\n", classes[l].c_str());
		}

		fclose(f);
		cout << f_name << endl;
	}
	cout << endl;

	for (unsigned int i = 0; i < n; i++) {
		sprintf(f_name, "%s_rebuilt_lda_%d.csv", f_label, i + 1);
		out_file = filesystem::current_path() / out_path / f_name;

		f = fopen(out_file.string().c_str(), "w");

		for (int l = 0; l < rebuilt_D[i].rows(); l++) {
			//fprintf(f, "%f", rebuilt_D[i](l, 0));
			for (int c = 0; c < rebuilt_D[i].cols(); c++) {
				fprintf(f, "%f,", rebuilt_D[i](l, c));
			}
			fprintf(f, "%s\n", classes[l].c_str());
		}

		fclose(f);
		cout << f_name << endl;
	}
	cout << endl;

	system("PAUSE");
	system("CLS");
}

void mhorvath::runPCA_LDAExperiment(const MatrixXd &X, const vector<string> &classes, const char * const f_label = "", const char * const out_path = "")
{
	unsigned const int n((unsigned int)X.cols()); // Number of features
	unsigned const int m((unsigned int)X.rows()); // Number of observations
	unsigned const int p_m((unsigned int)std::min((unsigned int)10, m)); // Limit of lines to print
	unsigned int p_n((unsigned int)std::min((unsigned int)10, n)); // Limit of lines to print
	FILE *f; // File used
	filesystem::path out_file; // Used to build output path
	char f_name[256]; // Used to build output file name
	MatrixXd reduction_pca(X.rows(), 2);
	MatrixXd reduction_lda(X.rows(), 2);

	if (strcmp(out_path, "")) {
		cout << filesystem::create_directory(out_path); // Create output path
	}

	cout << "######### " << f_label << " - PCA+LDA EXPERIMENT ########## " << endl << endl;

	mhorvath::PCA pca(X); // Compute PCA
	reduction_pca = pca.transform(3);

	cout << "Data =" << endl << X.block(0, 0, p_m, p_n) << endl << endl;
	cout << "Data Mean =" << endl << pca.getOriginalMean().segment(0, p_n) << endl << endl;
	cout << "DataAdjust =" << endl << pca.getDataAdjust().block(0, 0, p_m, p_n) << endl << endl;
	cout << "Covariance =" << endl << pca.covariance().block(0, 0, p_n, p_n) << endl << endl;
	cout << "Eigenvalues =" << endl << pca.values().segment(0, p_n) << endl << endl;
	cout << "Eigenvectors =" << endl << pca.components().block(0, 0, p_n, p_n) << endl << endl;
	cout << "Explained Variance =" << endl << pca.explained_variace_ratio().segment(0, p_n) << endl << endl;
	cout << "Sum of Explained Variance =" << endl << pca.explained_variace_ratio().sum() << endl << endl;

	mhorvath::LDA lda(reduction_pca, classes); // Compute LDA

	p_n = 2;
	reduction_lda = lda.transform(p_n);

	cout << "Data =" << endl << reduction_pca.block(0, 0, p_m, p_n) << endl << endl;
	cout << "Sb =" << endl << lda.getSb() << endl << endl;
	cout << "Sw =" << endl << lda.getSw() << endl << endl;
	cout << "Data Mean =" << endl << lda.data_mean() << endl << endl;
	cout << "Classes Mean =" << endl << lda.classes_mean() << endl << endl;
	cout << "Eigenvalues =" << endl << lda.values().segment(0, p_n) << endl << endl;
	cout << "Eigenvectors =" << endl << lda.components().block(0, 0, p_n, p_n) << endl << endl;
	cout << "Explained Variance =" << endl << lda.explained_variace_ratio().segment(0, p_n) << endl << endl;
	cout << "Sum of Explained Variance =" << endl << lda.explained_variace_ratio().sum() << endl << endl;

	cout << "GENERATED FILES: " << endl;

	sprintf(f_name, "%s_pca_lda.csv", f_label);
	out_file = filesystem::current_path() / out_path / f_name;

	f = fopen(out_file.string().c_str(), "w");

	for (int l = 0; l < reduction_lda.rows(); l++) {
		fprintf(f, "%f", reduction_lda(l, 0));
		for (int c = 1; c < reduction_lda.cols(); c++) {
			fprintf(f, ",%f", reduction_lda(l, c));
		}
		fprintf(f, "\n");
	}

	fclose(f);
	cout << f_name << endl;

	cout << endl;

	system("PAUSE");
	system("CLS");
}

double evaluateBetweenWithin(const MatrixXd &D, const mhorvath::KMeans &K) {
	const unsigned int m((unsigned int)D.rows());
	const unsigned int n((unsigned int)D.cols());
	const unsigned int k(K.getK());
	vector<unsigned int> classes(K.classifyMatrix(D));
	vector<unsigned int> count_m(k, 0);
	vector<RowVectorXd> centroids(K.getCentroids());
	RowVectorXd DataMean(D.colwise().mean());
	double Sb(0.0);
	double Sw(0.0);

	for (unsigned int i = 0; i < m; i++) {
		const unsigned int c(classes[i]);
		count_m[c]++;
		const RowVectorXd temp(D.row(i) - centroids[c]);

		Sw += temp * temp.transpose();
	}
	
	for (unsigned int i = 0; i < k; i++) {
		const RowVectorXd temp(centroids[i] - DataMean);
		
		Sb += temp * temp.transpose();
	}

	return Sb / Sw;
}

vector< vector<unsigned int> > confusionMatrix(const vector<unsigned int> * const original, const vector<unsigned int> &predicted, const unsigned int &k) {
	vector< vector<unsigned int> > c_matrix(k, vector<unsigned int>(k, 0));
	
	for (unsigned int i = 0; i < original->size(); i++) {
		c_matrix[original->at(i)][predicted[i]]++;
	}

	for (unsigned int i = 0; i < k; i++) {
		for (unsigned int j = 0; j < k; j++) {
			cout << c_matrix[i][j] << " ";
		}
		cout << endl;
	}

	return c_matrix;
}

void mhorvath::runKMeans_Experiment(const MatrixXd &X, const unsigned int &k, const char * const f_label, const char * const out_path, const std::vector<unsigned int> * const o_classes)
{
	const unsigned int n((unsigned int)X.cols()); // Number of features
	const unsigned int m((unsigned int)X.rows()); // Number of observations
	const unsigned int p_m((unsigned int)std::min((unsigned int)10, m)); // Limit of lines to print
	const unsigned int p_n((unsigned int)std::min((unsigned int)10, n)); // Limit of lines to print

	FILE *f; // File used
	filesystem::path out_file; // Used to build output path
	char f_name[256]; // Used to build output file name

	if (strcmp(out_path, "")) {
		filesystem::create_directory(out_path); // Create output path
	}

	cout << "######### " << f_label << " - k-MEANS EXPERIMENT ########## " << endl << endl;

	// Compute kmeans
	mhorvath::KMeans best_kmeans(X, k);
	double best_eval(evaluateBetweenWithin(X, best_kmeans));

	// Run k-Means 10000 times and keep the best evaluated result (Between-Within)
	for (unsigned int i = 0; i < 10000; i++) {
		const mhorvath::KMeans kmeans(X, k);
		const double eval(evaluateBetweenWithin(X, best_kmeans));

		if (eval > best_eval) {
			best_eval = eval;
			best_kmeans = kmeans;
		}
	}

	cout << "Between-Within Eval = " << best_eval << endl << endl;

	// Get centroids
	vector<RowVectorXd> centroids(best_kmeans.getCentroids());

	cout << "X =" << endl << X.block(0, 0, p_m, p_n) << endl << endl;

	// Print centroids
	cout << "Centroids:" << endl;

	for (unsigned int i = 0; i < centroids.size(); i++) {
		cout << centroids[i] << endl;
	}
	cout << endl;

	vector<unsigned int> classes(best_kmeans.classifyMatrix(X));

	// Print predicted classes
	cout << "Classifications (id-class)" << endl;

	for (unsigned int i = 0; i < classes.size(); i++) {
		printf("%d-%d ", i, classes[i]);
	}
	cout << endl << endl;
	
	vector< vector<unsigned int> > c_matrix;

	if (o_classes != 0) {
		// Print confusion matrix
		cout << "Confusion Matrix" << endl;

		c_matrix = confusionMatrix(o_classes, classes, k);
	}

	cout << "GENERATED FILES: " << endl;

	sprintf(f_name, "%s_kmeans_centroids.csv", f_label);
	out_file = filesystem::current_path() / out_path / f_name;

	f = fopen(out_file.string().c_str(), "w");

	for (unsigned int i = 0; i < k; i++) {
		fprintf(f, "%lf", centroids[i][0]);
		for (unsigned int j = 1; j < n; j++) {
			fprintf(f, ",%lf", centroids[i][j]);
		}
		fprintf(f, "\n");
	}

	fclose(f);
	cout << f_name << endl;

	sprintf(f_name, "%s_kmeans_classes.csv", f_label);
	out_file = filesystem::current_path() / out_path / f_name;

	f = fopen(out_file.string().c_str(), "w");

	for (unsigned int i = 0; i < m; i++) {
		fprintf(f, "%d\n", classes[i]);
	}

	fclose(f);
	cout << f_name << endl;

	if (o_classes != 0) {
		sprintf(f_name, "%s_confusion.csv", f_label);
		out_file = filesystem::current_path() / out_path / f_name;

		f = fopen(out_file.string().c_str(), "w");

		for (unsigned int i = 0; i < k; i++) {
			fprintf(f, "%d", c_matrix[i][0]);
			for (unsigned int j = 1; j < k; j++) {
				fprintf(f, ",%d", c_matrix[i][j]);
			}
			fprintf(f, "\n");
		}

		fclose(f);
		cout << f_name << endl;
	}

	cout << endl;

	system("PAUSE");
	system("CLS");
}

void mhorvath::inClassExample()
{
	// In class example: Some Data
	const unsigned int n(2); // Number of features
	const unsigned int m(10); // Number of observations
	MatrixXd D(m, n); // Data matrix
	MatrixXd X(m, n); // Data matrix
	VectorXd y(m); // Least squares target variable
	const char * const ex_label = "inClassExample"; // Experiment label
	const char * const output_folder = "pca_data"; // Output folder

	// Feature matrix X (mannualy defined)
	D << 2.5, 2.4,
		0.5, 0.7,
		2.2, 2.9,
		1.9, 2.2,
		3.1, 3.0,
		2.3, 2.7,
		2.0, 1.6,
		1.0, 1.1,
		1.5, 1.6,
		1.1, 0.9;

	// Initialize first column of feature matrix with 1
	X.col(0) = VectorXd::Zero(m).array() + 1.0;

	// Get feature matrix from data
	X.block(0, 1, m, n - 1) = D.block(0, 1, m, n - 1);

	// Get target variable from data
	y = D.col(n - 1);

	mhorvath::runPCAExperimentEx(D, ex_label, output_folder);
	mhorvath::runLeastSquaresExperiment(X, y, ex_label, output_folder);
}

void mhorvath::alpsWater()
{
	// In class example: Some Data
	const unsigned int n(2); // Number of features
	const unsigned int m(17); // Number of observations
	MatrixXd D(m, n); // Data matrix
	MatrixXd X(m, n); // Data matrix
	VectorXd y(m); // Least squares target variable
	const char * const ex_label = "alpsWater"; // Experiment label
	const char * const output_folder = "pca_data"; // Output folder
	char data[128];
	int temp; // Used to read useles dataset column
	FILE * f; // Used to read dataset file

	// Read dataset from file
	f = fopen("alpswater.txt", "r");

	fgets(data, 126, f); // Remove header line

	// Read line-by-line
	for (int i = 0; i < m; i++) {
		fscanf(f, "%d %lf %lf", &temp, &D(i, 0), &D(i, 1));
	}

	fclose(f); // Close file

	// Initialize first column of feature matrix with 1
	X.col(0) = VectorXd::Zero(m).array() + 1.0;

	// Get feature matrix from data
	X.col(1) = D.col(1);

	// Get target variable from data
	y = D.col(0);

	mhorvath::runPCAExperimentEx(D, ex_label, output_folder);
	mhorvath::runLeastSquaresExperiment(X, y, ex_label, output_folder);
}

void mhorvath::booksXgrades()
{
	// In class example: Some Data
	const unsigned int n(3); // Number of features
	const unsigned int m(40); // Number of observations
	MatrixXd D(m, n); // Data matrix
	MatrixXd X(m, n); // Data matrix
	VectorXd y(m); // Least squares target variable
	const char * const ex_label = "booksXgrades"; // Experiment label
	const char * const output_folder = "pca_data"; // Output folder
	FILE * f; // Used to read dataset file

	// Read dataset from file
	f = fopen("Books_attend_grade.txt", "r");

	// Read line-by-line
	for (int i = 0; i < m; i++) {
		fscanf(f, "%lf %lf %lf", &D(i, 0), &D(i, 1), &D(i, 2));
	}

	fclose(f); // Close file

	// Initialize first column of feature matrix with 1
	X.col(0) = VectorXd::Zero(m).array() + 1.0;

	// Get feature matrix from data
	X.block(0, 1, m, n - 1) = D.block(0, 0, m, n - 1);

	// Get target variable from data
	y = D.col(2);

	mhorvath::runPCAExperimentEx(D, ex_label, output_folder);
	mhorvath::runLeastSquaresExperiment(X, y, ex_label, output_folder);
}

void mhorvath::usCensus()
{
	// In class example: Some Data
	const unsigned int n(2); // Number of features
	const unsigned int m(11); // Number of observations
	MatrixXd D(m, n); // Data matrix
	MatrixXd X(m, n); // Data matrix
	VectorXd y(m); // Least squares target variable
	const char * const ex_label = "usCensus"; // Experiment label
	const char * const output_folder = "pca_data"; // Output folder
	FILE * f; // Used to read dataset file

	// Read dataset from file
	f = fopen("usCensus.txt", "r");

	// Read line-by-line
	for (int i = 0; i < m; i++) {
		fscanf(f, "%lf,%lf", &D(i, 0), &D(i, 1));
	}

	fclose(f); // Close file

	// Initialize first column of feature matrix with 1
	X.col(0) = VectorXd::Zero(m).array() + 1.0;

	// Get feature matrix from data
	X.col(1) = D.col(0);

	// Get target variable from data
	y = D.col(1);

	mhorvath::runPCAExperimentEx(D, ex_label, output_folder);
	mhorvath::runLeastSquaresExperiment(X, y, ex_label, output_folder);
}

void mhorvath::hald()
{
	// In class example: Some Data
	const unsigned int n(4); // Number of features
	const unsigned int m(13); // Number of observations
	MatrixXd D(m, n); // Data matrix
	//MatrixXd X(m, n); // Data matrix
	VectorXd y(m); // Least squares target variable
	const char * const ex_label = "hald"; // Experiment label
	FILE * f; // Used to read dataset file
	char data[128];

	// Read dataset from file
	f = fopen("hald.txt", "r");

	fgets(data, 126, f); // Remove header line

	// Read line-by-line
	for (int i = 0; i < m; i++) {
		fscanf(f, "%lf %lf %lf %lf %lf", &y(i), &D(i, 0), &D(i, 1), &D(i, 2), &D(i, 3));
	}

	fclose(f); // Close file

	//// Initialize first column of feature matrix with 1
	//X.col(0) = VectorXd::Zero(m).array() + 1.0;

	//// Get feature matrix from data
	//X.col(1) = D.col(0);

	//// Get target variable from data
	//y = D.col(1);

	mhorvath::PCA pca(D);

	cout << "Eigenvectors =" << endl << pca.components() << endl << endl;
	cout << "Eigenvalues =" << endl << pca.values() << endl << endl;
	cout << "Explained Variance =" << endl << pca.explained_variace_ratio() << endl << endl;
	cout << "Cumulative Explained Variance =" << endl;

	double sum_variance(0.0);

	for (int i = 0; i < pca.explained_variace_ratio().size(); i++) {
		sum_variance += pca.explained_variace_ratio()(i);
		printf("With %d principal component: %.3f\n", i + 1, sum_variance);
	}

	cout << endl << endl;


	system("PAUSE");
}

void mhorvath::iris()
{
	// In class example: Some Data
	const unsigned int n(4); // Number of features
	const unsigned int m(150); // Number of observations
	const unsigned int p_n(min(n, (unsigned int)10)); // Number of features
	const unsigned int p_m(min(m, (unsigned int)10)); // Number of observations
	const unsigned int k(3); // Number of classes
	MatrixXd D(m, n); // Data matrix
	vector<unsigned int> classes(m); // Least squares target variable
	const char * const ex_label = "iris"; // Experiment label
	const char * const output_folder = "kmeans_data"; // Output folder
	FILE * f; // Used to read dataset file
	//char data[128]; // Used to read class variable

	// Read dataset from file
	f = fopen("iris.txt", "r");

	// Read line-by-line
	for (int i = 0; i < m; i++) {
		fscanf(f, "%lf,%lf,%lf,%lf,%d\n", &D(i, 0), &D(i, 1), &D(i, 2), &D(i, 3), &classes[i]);
		classes[i]--;
		//classes[i] = data;
	}

	fclose(f); // Close file

	//mhorvath::runLDAExperiment(D, classes, ex_label, output_folder);
	//mhorvath::runPCAExperimentEx(D, ex_label, output_folder);
	//mhorvath::runPCA_LDAExperiment(D, classes, ex_label, output_folder);
	mhorvath::runKMeans_Experiment(D, k, ex_label, output_folder, &classes);
}

void mhorvath::inClassExampleKMeans()
{
	// In class example: k-Means exercise 1
	const unsigned int n(2); // Number of features
	const unsigned int m(17); // Number of observations
	const unsigned int p_n(min(n, (unsigned int)10)); // Number of features
	const unsigned int p_m(min(m, (unsigned int)10)); // Number of observations
	const unsigned int k(3); // Number of classes
	MatrixXd D(m, n); // Data matrix
	vector<string> classes(m); // Least squares target variable
	const char * const ex_label = "inClassExample"; // Experiment label
	const char * const output_folder = "kmeans_data"; // Output folder
	FILE * f; // Used to read dataset file
	char data[128]; // Used to read class variable

	// Read dataset from file
	f = fopen("inClassExample_KMeans.txt", "r");

	// Read line-by-line
	for (int i = 0; i < m; i++) {
		fscanf(f, "%lf,%lf\n", &D(i, 0), &D(i, 1));
		classes[i] = data;
	}

	fclose(f); // Close file

	mhorvath::runKMeans_Experiment(D, k, ex_label, output_folder);
}

void mhorvath::seedsUCI()
{
	// UCI Breast Cancer Winsconsin (Diagnostic) Data Set
	const unsigned int n(7); // Number of features
	const unsigned int m(210); // Number of observations
	const unsigned int p_n(min(n, (unsigned int)10)); // Number of features
	const unsigned int p_m(min(m, (unsigned int)10)); // Number of observations
	const unsigned int k(3); // Number of classes
	MatrixXd D(m, n); // Data matrix
	vector<unsigned int> classes(m); // Least squares target variable
	const char * const ex_label = "seeds"; // Experiment label
	const char * const output_folder = "kmeans_data"; // Output folder
	FILE * f; // Used to read dataset file
	//char data[128]; // Used to read class variable

	// Read dataset from file
	f = fopen("seeds.txt", "r");

	// Read line-by-line
	for (unsigned int i = 0; i < m; i++) {
		fscanf(f, "%lf", &D(i, 0));

		// Read feature-by-feature
		for (unsigned int j = 1; j < n; j++) {
			fscanf(f, ",%lf", &D(i, j));
		}

		fscanf(f, ",%d\n", &classes[i]);
		classes[i]--;
	}

	fclose(f); // Close file

	mhorvath::runKMeans_Experiment(D, k, ex_label, output_folder, &classes);
}

void mhorvath::winesUCI()
{
	// UCI Breast Cancer Winsconsin (Diagnostic) Data Set
	const unsigned int n(13); // Number of features
	const unsigned int m(178); // Number of observations
	const unsigned int p_n(min(n, (unsigned int)10)); // Number of features
	const unsigned int p_m(min(m, (unsigned int)10)); // Number of observations
	const unsigned int k(3); // Number of classes
	MatrixXd D(m, n); // Data matrix
	vector<unsigned int> classes(m); // Least squares target variable
	const char * const ex_label = "wine"; // Experiment label
	const char * const output_folder = "kmeans_data"; // Output folder
	FILE * f; // Used to read dataset file
	//char data[128]; // Used to read class variable

	// Read dataset from file
	f = fopen("wine.txt", "r");

	// Read line-by-line
	for (unsigned int i = 0; i < m; i++) {
		fscanf(f, "%d", &classes[i]);
		classes[i]--;

		// Read feature-by-feature
		for (unsigned int j = 0; j < n; j++) {
			fscanf(f, ",%lf", &D(i, j));
		}
	}

	fclose(f); // Close file

	mhorvath::runKMeans_Experiment(D, k, ex_label, output_folder, &classes);
}

void mhorvath::inClassExampleLDA()
{
	// In class example: Some Data
	const unsigned int n(2); // Number of features
	const unsigned int m(11); // Number of observations
	MatrixXd D(m, n); // Data matrix
	vector<string> classes(m); // Least squares target variable
	const char * const ex_label = "inClassExample"; // Experiment label
	const char * const output_folder = "lda_data"; // Output folder
	FILE * f; // Used to read dataset file
	char data[128];

	// Read dataset from file
	f = fopen("inClassExample_LDA.txt", "r");

	// Read line-by-line
	for (int i = 0; i < m; i++) {
		fscanf(f, "%lf %lf %s\n", &D(i, 0), &D(i, 1), &data);
		classes[i] = data;
	}

	fclose(f); // Close file

	mhorvath::runLDAExperiment(D, classes, ex_label, output_folder);
	mhorvath::runPCAExperimentEx(D, ex_label, output_folder);
}
