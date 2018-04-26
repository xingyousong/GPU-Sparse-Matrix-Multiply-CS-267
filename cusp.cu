#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <cusp/csr_matrix.h>
#include <cusp/multiply.h>
#include <cusp/galley/random.h>
#include <cusp/print.h>
#include <iostream>
#include <random>

#define P_SIZE 14 

std::map<double, std::map<int, double>> my_map;

const double p[] = {0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5};

void writeToCsv(std::string fname) {
	ofstream myFile;
	myFile.open(fname);
	myFile << ",16,32,64,128,256,512,1024,2048,4096,8192,16384\n"
	for(int i = 0; i < P_SIZE; i++) {
		std::map<int, double> inner_map = my_map[p[i]];
		myFile << p[i];
		myFile << ",";
		std::map<int, double>::iterator it;
		std::map<int, double>::iterator next;
		for(it = inner_map.begin(); it != inner_map.end(); it++) {
			myFile << it->second;
			next = it + 1;
			if(next != inner_map.end()) {
				myFile << ",";
			} else {
				myFile << "\n";
			}
		}
	}
	myFile.close();
}

int csrTimesCsr() {
	for (int i = 4; i < 15; i++) {
		int N = pow(2, i);
		for (int j = 0; j < P_SIZE; j++) {
			cusp::csr_matrix<int, int, cusp::device_memory> X_GPU;
			cusp::csr_matrix<int, int, cusp::device_memory> Y_GPU;
			int num_samples = ((int) p[j] * N * N);
			cusp::gallery::random(N, N, num_samples, X_GPU);
			cusp::gallery::random(N, N, num_samples, Y_GPU);
			cusp::array2d<int, cusp::device_memory> Z(N,N);
			clock_t start = clock();
			cusp::multiply(X_GPU, Y_GPU, Z);
			clock_t diff = clock() - start;
			double sec = ((double) diff) / CLOCKS_PER_SEC;
			if(my_map.count(p[j]) == 0) {
				my_map[p[j]] = {N: sec};
			} else {
				my_map[p[j]][N] = sec;
			}
		}
	}
	return 1;
}

int main() {
	csrTimesCsr();
	writeToCsv("cusp_sparseCSR_sparseCSR.csv");
}