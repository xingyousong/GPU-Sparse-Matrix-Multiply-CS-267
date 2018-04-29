#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <cusp/csr_matrix.h>
#include <cusp/array2d.h>
#include <cusp/array1d.h>
#include <cusp/multiply.h>
#include <cusp/gallery/random.h>
#include <cusp/gallery/poisson.h>
#include <cusp/functional.h>
#include <cusp/print.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <random>
#include <map>

#define P_SIZE 14

using namespace std;

thrust::identity<int> zero;
thrust::multiplies<int> combine;
thrust::plus<int> reduce;
ofstream outputFile;

const double p[] = {0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5};

int csrTimesCsr() {
        outputFile.open("cusp_csvs/cusp_csr_csr.csv");
        for (int i = 4; i < 15; i++) {
                int N = pow(2, i);
                for (int j = 0; j < P_SIZE; j++) {
                        cusp::csr_matrix<int, int, cusp::device_memory> X_GPU;
                        cusp::csr_matrix<int, int, cusp::device_memory> Y_GPU;
                        int num_samples = ((int) (p[j] * N * N));
                        cout << "N is: " << N << " density is " << p[j];
                        cout << "\nNum Samples: " << num_samples << "\n";
                        cusp::gallery::random(X_GPU, N, N, num_samples);
                        cusp::gallery::random(Y_GPU, N, N, num_samples);
                        cusp::csr_matrix<int, int, cusp::device_memory> Z;
                        clock_t start = clock();
                        cusp::generalized_spgemm(X_GPU, Y_GPU, Z, zero, combine, reduce);
                        clock_t diff = clock() - start;
                        float sec = ((float) diff) / CLOCKS_PER_SEC;
                        outputFile << N << "," << p[j] << "," << std::setprecision(15) << sec << "\n";
                }
        }
        outputFile.close();
        return 1;
}

int csrTimesDenseVector() {
         ofstream outputFile;
         outputFile.open("cusp_csvs/cusp_csr_denseVec.csv");
         for(int i = 4; i < 15; i++) {
                 int N = pow(2, i);
                 for(int j = 0; j < P_SIZE; j++) {
                     cusp::csr_matrix<int, int, cusp::device_memory> X_GPU;
                     cusp::csr_matrix<int, int, cusp::device_memory> Y_GPU;
                     int num_samples = ((int) (p[j] * N * N));
                     int randomNum = rand() % N;
                     cout << "N is: " << N << " density is " << p[j];
                     cout << "\nNum Samples: " << num_samples << "\n";
                     cusp::gallery::random(X_GPU, N, N, num_samples);
                     cusp::gallery::random(Y_GPU, N, 1, randomNum);
                     cusp::csr_matrix<int, int, cusp::device_memory> Z;
                     clock_t start = clock();
                     cusp::generalized_spgemm(X_GPU, Y_GPU, Z, zero, combine, reduce);
                     clock_t diff = clock() - start;
                     float sec = ((float) diff) / CLOCKS_PER_SEC;
                     outputFile << N << "," << p[j] << "," << std::setprecision(15) << sec << "\n";
                 }
         }
         outputFile.close();
         return 1;
}

int csrTimesDenseMatrix() {
    typedef cusp::csr_matrix<int, float, cusp::device_memory>             Matrix;
    typedef cusp::array1d<float, cusp::device_memory>                     Array1d;
    typedef cusp::array2d<float, cusp::device_memory, cusp::column_major> Array2d;
    typedef Array2d::column_view column_view;

    ofstream outputFile;
    outputFile.open("cusp_csvs/cusp_csr_denseMat.csv");

    for (int i = 4; i < 15; i++) {
        int N = pow(2, i);
        for(int j = 0; j < P_SIZE; j++) {
            Matrix A;
            int num_samples = ((int) (p[j] * N * N));
            int randomNum = rand() % N;
            cusp::gallery::random(A, N, N, num_samples);

            Array2d X(N, N);
            cusp::gallery::random(X, N, N, randomNum);
            Array1d y(N);

            column_view x = X.column(0);
            cout << "N is: " << N << " density is " << p[j];
            cout << "\nNum Samples: " << num_samples << "\n";
            clock_t start = clock();
            cusp::multiply(A, x, y);
            clock_t diff = clock() - start;
            float sec = ((float) diff) / CLOCKS_PER_SEC;
            outputFile << N << "," << p[j] << "," << std::setprecision(15) << sec << "\n";
        }
    }
    return 1;
}




int main() {
        csrTimesCsr();
        csrTimesDenseVector();
        csrTimesDenseMatrix();
        system("sudo shutdown -P now");
}
