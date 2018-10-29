#include "assignments.h"
#include "pca.h"
#include "least_squares.h"
#include "lda.h"
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
	unsigned const int p_m((unsigned int)std::min((unsigned int) 10, m)); // Limit of lines to print
	unsigned const int p_n((unsigned int)std::min((unsigned int) 10, n)); // Limit of lines to print
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
	MatrixXd D(m, n); // Data matrix
	vector<string> classes(m); // Least squares target variable
	const char * const ex_label = "iris"; // Experiment label
	FILE * f; // Used to read dataset file
	char data[128];

	// Read dataset from file
	f = fopen("iris.txt", "r");

	// Read line-by-line
	for (int i = 0; i < m; i++) {
		fscanf(f, "%lf,%lf,%lf,%lf,%s\n", &D(i, 0), &D(i, 1), &D(i, 2), &D(i, 3), &data);
		classes[i] = data;
	}

	fclose(f); // Close file

	mhorvath::PCA pca(mhorvath::lda(D, classes));

	cout << "Data =" << endl << D.block(0, 0, p_m, p_n) << endl << endl;
	cout << "Data Mean =" << endl << pca.getOriginalMean().segment(0, p_n) << endl << endl;
	cout << "DataAdjust =" << endl << pca.getDataAdjust().block(0, 0, p_n, p_n) << endl << endl;
	cout << "Covariance =" << endl << pca.covariance().block(0, 0, p_n, p_n) << endl << endl;
	cout << "Eigenvalues =" << endl << pca.values().segment(0, p_n) << endl << endl;
	cout << "Eigenvectors =" << endl << pca.components().block(0, 0, p_n, p_n) << endl << endl;
	cout << "Explained Variance =" << endl << pca.explained_variace_ratio().segment(0, p_n) << endl << endl;
	cout << "Sum of Explained Variance =" << endl << pca.explained_variace_ratio().sum() << endl << endl;

	system("PAUSE");
}
