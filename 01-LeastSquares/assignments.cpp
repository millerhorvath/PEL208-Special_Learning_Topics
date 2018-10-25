#include "assignments.h"

using namespace std;
using namespace Eigen;

void mhorvath::inClassExample()
{
	// In class example: Height x Shoe Size
	MatrixXd X(10, 2); // Feature matrix
	VectorXd y(10); // Target vector
	VectorXd B[3]; // Coefficients vector (Beta)
	VectorXd W[3]; // Weight vector
	VectorXd x(2); // Feature vector
	double pred_y[3]; // Predicted target values
	FILE * f; // Used to write the computed coefficients in file

	// Feature matrix X (mannualy defined)
	X << 1, 69,
		1, 67,
		1, 71,
		1, 65,
		1, 72,
		1, 68,
		1, 74,
		1, 65,
		1, 66,
		1, 72;

	// Target values y (mannualy defined)
	y << 9.5, 8.5, 11.5, 10.5, 11, 7.5, 12, 7, 7.5, 13;

	// Print X and y
	cout << "X (10 x 2 matrix) =" << endl << X << endl << endl;
	cout << "y (10-dimensional vector)=" << endl << y << endl << endl;


	// Pause terminal for data vizualization --------------------
	system("PAUSE");
	system("CLS");


	// Call least squares function and store coefficients in Vector B
	B[0] = mhorvath::leastSquares(X, y);

	// Print coefficients 
	cout << "Least Squares coefficients B =" << endl << B[0] << endl << endl;


	// Pause terminal for data vizualization --------------------
	system("PAUSE");
	system("CLS");


	// Compute weight vectors
	W[0] = VectorXd(ArrayXd::Zero(y.size()) + 1.0);
	W[1] = VectorXd(ArrayXd::Zero(y.size()) + 1.0);
	W[1](3) = W[1](5) = W[1](9) = 0.0;
	W[2] = VectorXd(1.0 / (y - X * B[0]).cwiseAbs().array());

	// Print weight vector
	cout << "W1 (w_i = 1) =" << endl << W[0] << endl << endl;
	cout << "W2 (w_i = selective) =" << endl << W[1] << endl << endl;
	cout << "W3 (w_i = 1 / (y - X*B)) =" << endl << W[2] << endl << endl;


	// Pause terminal for data vizualization --------------------
	system("PAUSE");
	system("CLS");


	// Compute weighted least squares using in class weight vector examples
	for (int i = 0; i < 3; i++) {
		B[i] = mhorvath::weightedLeastSquares(X, y, W[i]);
	}


	// Print weghted coefficients
	cout << "Weighted Least Squares coefficients B (1) =" << endl << B[0] << endl << endl;
	cout << "Weighted Least Squares coefficients B (selective) =" << endl << B[1] << endl << endl;
	cout << "Weighted Least Squares coefficients B (1 / (y - X*B)) =" << endl << B[2] << endl << endl;

	// Compute prediction for x = 67
	x << 1, 67;


	for (int i = 0; i < 3; i++) {
		pred_y[i] = x.transpose() * B[i];
	}

	// Print predicted y for x = 67
	cout << "y(67) (1) =" << endl << pred_y[0] << endl << endl;
	cout << "y(67) (selective) =" << endl << pred_y[1] << endl << endl;
	cout << "y(67) (1 / (y - X*B)) =" << endl << pred_y[2] << endl << endl;


	// Pause terminal for data vizualization --------------------
	system("PAUSE");
	system("CLS");

	f = fopen("inClassExample_coefs.csv", "w");

	fprintf(f, "b,a\n");

	for (int i = 0; i < 3; i++) {
		fprintf(f, "%f", B[i](0));
		for (int j = 1; j < B[i].size(); j++) {
			fprintf(f, ",%f", B[i](j));
		}
		fprintf(f, "\n");
	}

	fclose(f);
}

void mhorvath::usCensus()
{
	MatrixXd X(11, 3); // Feature Matrix
	VectorXd y(11); // Target vector
	VectorXd B[8]; // Coefficients vector (Beta)
	VectorXd x[2]; // Feature vector
	FILE *f; // Used to write the computed coefficients in file

	double pred_population[8]; // Predicted population

	// Initialize x vectors
	x[0] = VectorXd(2);
	x[1] = VectorXd(3);


	X << 1, 1900, 1900 * 1900, 1, 1910, 1910 * 1910, 1, 1920, 1920 * 1920, 1, 1930, 1930 * 1930,
		1, 1940, 1940 * 1940, 1, 1950, 1950 * 1950, 1, 1960, 1960 * 1960, 1, 1970, 1970 * 1970,
		1, 1980, 1980 * 1980, 1, 1990, 1990 * 1990, 1, 2000, 2000 * 2000;

	y << 75.9950, 91.9720, 105.7110, 123.2030, 131.6690,
		150.6970, 179.3230, 203.2120, 226.5050, 249.6330, 281.4220;

	// Print X and y
	cout << "X (11 x 3 matrix) =" << endl << X << endl << endl;
	cout << "y (11-dimensional vector)=" << endl << y << endl << endl;


	// Pause terminal for data vizualization --------------------
	system("PAUSE");
	system("CLS");


	// Compute least squares
	B[0] = mhorvath::leastSquares(X.block(0, 0, X.rows(), 2), y); // Fit data into a line
	B[1] = mhorvath::leastSquares(X, y); // Fit data into a quadratic function

	// Compute weighted least squares
	B[2] = mhorvath::weightedLeastSquares(X.block(0, 0, X.rows(), 2), y); // Fit data into a line
	B[3] = mhorvath::weightedLeastSquares(X, y); // Fit data into a quadratic function

	// Compute least squares with pseudo-inverse
	B[4] = mhorvath::leastSquares_pinv(X.block(0, 0, X.rows(), 2), y); // Fit data into a line
	B[5] = mhorvath::leastSquares_pinv(X, y); // Fit data into a quadratic function

	// Compute weighted least squares with pseudo-inverse
	B[6] = mhorvath::weightedLeastSquares_pinv(X.block(0, 0, X.rows(), 2), y); // Fit data into a line
	B[7] = mhorvath::weightedLeastSquares_pinv(X, y); // Fit data into a quadratic function

	// Print coefficients
	cout << "B1 =" << endl << B[0] << endl << endl;
	cout << "B2 =" << endl << B[1] << endl << endl;
	cout << "B3 =" << endl << B[2] << endl << endl;
	cout << "B4 =" << endl << B[3] << endl << endl;

	// Predict population in 2010
	x[0] << 1.0, 2010;
	x[1] << 1.0, 2010, 2010 * 2010;

	for (int i = 0; i < 8; i++) {
		pred_population[i] = x[i % 2].transpose() * B[i];
	}

	// Print prediction
	for (int i = 0; i < 8; i++) {
		cout << "y" << i + 1 << "(2010) =" << endl << pred_population[i] << endl << endl;
	}

	// Pause terminal for data vizualization --------------------
	system("PAUSE");
	system("CLS");

	f = fopen("usCensus_coefs.csv", "w");

	fprintf(f, "c,b,a\n");

	for (int i = 0; i < 8; i++) {
		fprintf(f, "%f", B[i](0));

		for (int j = 1; j < B[i].size(); j++) {
			fprintf(f, ",%f", B[i](j));
		}

		if (i % 2 == 0) {
			fprintf(f, ",%f", 0.0);
		}
		fprintf(f, "\n");
	}

	fclose(f);
}

void mhorvath::boilingPointsAtAlps()
{
	MatrixXd X(17, 3); // Feature Matrix
	VectorXd y(17); // Target vector
	VectorXd B[8]; // Coefficients vector (Beta)
	VectorXd x[2]; // Feature vector

	double pred_pressure[8]; // Predicted pressure
	double boiling[17]; // Vetor used to read boiling feature values from file
	double pressure[17]; // Vetor used to read pressure target values from file
	int n(17); // Dataset size
	int row[17]; // Vetor used to read row feature values from file
	char data[512]; // Char string to read dataset header
	FILE *f; // Used to read dataset file and write the computed coefficients in file

	// Initialize x vectors
	x[0] = VectorXd(2);
	x[1] = VectorXd(3);

	// Read dataset from file
	f = fopen("alpswater.txt", "r");
	
	// Remove header line
	fgets(data, 511, f);

	// Read line-by-line
	for (int i = 0; i < n; i++) {
		fscanf(f, "%d %lf %lf", &row[i], &pressure[i], &boiling[i]);
	}

	// Close file
	fclose(f);

	// Store data into features matrix and target vector
	for (int i = 0; i < n; i++) {
		X(i, 0) = 1;
		X(i, 1) = boiling[i];
		X(i, 2) = boiling[i] * boiling[i];
		y(i) = pressure[i];
	}

	// Print X and y
	cout << "10 first observations in X (17 x 3 matrix) =" << endl << X.block(0, 0, 10, X.cols()) << endl << endl;
	cout << "10 first observations in y (17-dimensional vector)=" << endl << y.segment(0, 10) << endl << endl;


	// Pause terminal for data vizualization --------------------
	system("PAUSE");
	system("CLS");


	// Compute least squares
	B[0] = mhorvath::leastSquares(X.block(0, 0, X.rows(), 2), y); // Fit data into a line
	B[1] = mhorvath::leastSquares(X, y); // Fit data into a quadratic function
	
	// Compute weighted least squares
	B[2] = mhorvath::weightedLeastSquares(X.block(0, 0, X.rows(), 2), y); // Fit data into a line
	B[3] = mhorvath::weightedLeastSquares(X, y); // Fit data into a quadratic function

	// Compute least squares with pseudo-inverse
	B[4] = mhorvath::leastSquares_pinv(X.block(0, 0, X.rows(), 2), y); // Fit data into a line
	B[5] = mhorvath::leastSquares_pinv(X, y); // Fit data into a quadratic function

	// Compute weighted least squares with pseudo-inverse
	B[6] = mhorvath::weightedLeastSquares_pinv(X.block(0, 0, X.rows(), 2), y); // Fit data into a line
	B[7] = mhorvath::weightedLeastSquares_pinv(X, y); // Fit data into a quadratic function

	// Print coefficients
	cout << "B1 =" << endl << B[0] << endl << endl;
	cout << "B2 =" << endl << B[1] << endl << endl;
	cout << "B3 =" << endl << B[2] << endl << endl;
	cout << "B4 =" << endl << B[3] << endl << endl;

	// Predict pressure at 200.0 F
	x[0] << 1.0, 200.0;
	x[1] << 1.0, 200.0, 200.0*200.0;

	for (int i = 0; i < 8; i++) {
		pred_pressure[i] = x[i%2].transpose() * B[i];
	}

	// Print prediction
	for (int i = 0; i < 8; i++) {
		cout << "y" << i+1 << "(200.0) =" << endl << pred_pressure[i] << endl << endl;
	}

	// Pause terminal for data vizualization --------------------
	system("PAUSE");
	system("CLS");

	f = fopen("boilingPointsAtAlps_coefs.csv", "w");

	fprintf(f, "c,b,a\n");

	for (int i = 0; i < 8; i++) {
		fprintf(f, "%f", B[i](0));

		for (int j = 1; j < B[i].size(); j++) {
			fprintf(f, ",%f", B[i](j));
		}

		if (i % 2 == 0) {
			fprintf(f, ",%f", 0.0);
		}
		fprintf(f, "\n");
	}

	fclose(f);
}

void mhorvath::booksXgrades()
{
	MatrixXd X(40, 5); // Feature Matrix
	VectorXd y(40); // Target vector
	VectorXd B[8]; // Coefficients vector (Beta)
	VectorXd x[2]; // Feature vector

	double pred_grade[8]; // Predicted grade
	float books[40]; // Vetor used to read books feature values from file
	float attend[40]; // Vetor used to read attend feature values from file
	float grade[40]; // Vetor used to read grade target values from file
	int n(40); // Dataset size
	FILE *f; // Used to read dataset file

	// Initialize x vectors
	x[0] = VectorXd(3);
	x[1] = VectorXd(5);

	// Read dataset from file
	f = fopen("Books_attend_grade.dat", "r");

	// Read line-by-line
	for (int i = 0; i < n; i++) {
		fscanf(f, "%f %f %f", &books[i], &attend[i], &grade[i]);
	}

	// Close file
	fclose(f);

	// Store data into features matrix and target vector
	for (int i = 0; i < n; i++) {
		X(i, 0) = 1;
		X(i, 1) = books[i];
		X(i, 2) = attend[i];
		X(i, 3) = books[i] * books[i];
		X(i, 4) = attend[i] * attend[i];
		y(i) = grade[i];
	}

	// Print X and y
	cout << "10 first observations in X (40 x 5 matrix) =" << endl << X.block(0, 0, 10, X.cols()) << endl << endl;
	cout << "10 first observations in y (40-dimensional vector) =" << endl << y.segment(0, 10) << endl << endl;


	// Pause terminal for data vizualization --------------------
	system("PAUSE");
	system("CLS");


	// Compute least squares
	B[0] = mhorvath::leastSquares(X.block(0, 0, X.rows(), 3), y); // Fit data into a line
	B[1] = mhorvath::leastSquares(X, y); // Fit data into a quadratic function

	// Compute weighted least squares
	B[2] = mhorvath::weightedLeastSquares(X.block(0, 0, X.rows(), 3), y); // Fit data into a line
	B[3] = mhorvath::weightedLeastSquares(X, y); // Fit data into a quadratic function

	// Compute least squares with pseudo-inverse
	B[4] = mhorvath::leastSquares_pinv(X.block(0, 0, X.rows(), 3), y); // Fit data into a line
	B[5] = mhorvath::leastSquares_pinv(X, y); // Fit data into a quadratic function

	// Compute weighted least squares with pseudo-inverse
	B[6] = mhorvath::weightedLeastSquares_pinv(X.block(0, 0, X.rows(), 3), y); // Fit data into a line
	B[7] = mhorvath::weightedLeastSquares_pinv(X, y); // Fit data into a quadratic function

	// Print coefficients
	cout << "B1 =" << endl << B[0] << endl << endl;
	cout << "B2 =" << endl << B[1] << endl << endl;
	cout << "B3 =" << endl << B[2] << endl << endl;
	cout << "B4 =" << endl << B[3] << endl << endl;

	// Predict grade for books = 2 and attend = 16
	x[0] << 1.0, 2.0, 16.0;
	x[1] << 1.0, 2.0, 16.0, 2.0*2.0, 16.0*16.0;
	
	for (int i = 0; i < 8; i++) {
		pred_grade[i] = x[i%2].transpose() * B[i];
	}

	// Print prediction
	for (int i = 0; i < 8; i++) {
		cout << "y(2, 16) =" << endl << pred_grade[i] << endl << endl;
	}

	// Pause terminal for data vizualization --------------------
	system("PAUSE");
	system("CLS");

	f = fopen("booksXgrades_coefs.csv", "w");

	fprintf(f, "c,b1,b2,a1,a2\n");

	for (int i = 0; i < 8; i++) {
		fprintf(f, "%f", B[i](0));

		for (int j = 1; j < B[i].size(); j++) {
			fprintf(f, ",%f", B[i](j));
		}

		if (i % 2 == 0) {
			for (int j = 0; j < 2; j++) {
				fprintf(f, ",%f", 0.0);
			}
		}
		fprintf(f, "\n");
	}

	fclose(f);
}
