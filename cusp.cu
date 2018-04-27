#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <cusp/csr_matrix.h>
#include <cusp/multiply.h>
#include <cusp/gallery/random.h>
#include <cusp/print.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <random>
#include <map>

#define P_SIZE 14

using namespace std;

const double p[] = {0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5};

int csrTimesCsr() {
        ofstream outputFile;
        outputFile.open("cusp_csr_csr.csv");
        for (int i = 4; i < 15; i++) {
                int N = pow(2, i);
                for (int j = 0; j < P_SIZE; j++) {
                        cusp::csr_matrix<int, int, cusp::device_memory> X_GPU;
                        cusp::csr_matrix<int, int, cusp::device_memory> Y_GPU;
                        int num_samples = ((int) (p[j] * N * N));
                        cout << "N is: " << N << " density is " << p[j];
                        cout << "\nNum Samples: " << num_samples << "\n";
                        cusp::gallery::random(N, N, num_samples, X_GPU);
                        cusp::gallery::random(N, N, num_samples, Y_GPU);
                        cusp::csr_matrix<int, int, cusp::device_memory> Z;
                        clock_t start = clock();
                        cusp::multiply(X_GPU, Y_GPU, Z);
                        clock_t diff = clock() - start;
                        float sec = ((float) diff) / CLOCKS_PER_SEC;
                        outputFile << N << "," << p[j] << "," << std::setprecision(15) << sec << "\n";
                }
        }
        return 1;
}

int main() {
        csrTimesCsr();
        //system("shutdown -s");
}
