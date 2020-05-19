#include <fstream>
#include <iostream>
#include <chrono> 
# include <mutex>
#include <stdio.h>
#include <string.h>


using namespace std;
# define NO_OF_CHARS 256  
// preprocessing function 
__device__ void badCharHeuristic(char* str, int size,
	int badchar[NO_OF_CHARS]) {
	int i;

	// Initialize all occurrences as -1  
	for (i = 0; i < NO_OF_CHARS; i++)
		badchar[i] = -1;

	// Fill the actual value of last occurrence  
	for (i = 0; i < size; i++)
		badchar[(int)str[i]] = i;
}

//implementation Boyer Moore Algorithm
__global__ void search(const char* txt, char* pat, int ns, int ms, int * t) {

	int i = threadIdx.x;
	int start = i * ns;
	int m = ms;
	int n = ns;

	int badchar[NO_OF_CHARS];
	/* Fill the bad character array by calling
	the preprocessing function badCharHeuristic()
	for given pattern */
	badCharHeuristic(pat, m, badchar);

	int s = 0; // s = shift of pattern
	while (s <= (n - m)) {
		int j = m - 1;
		while (j >= 0 && pat[j] == txt[s + j + start]) {
			j--;
		}
		if (j < 0) {
			//cout << "found pattern at = " << s << endl;
			atomicAdd(t, 1);
			s += (s + m < n) ? m - badchar[txt[s + m + start]] : 1;
		}
		else {
			int a = j - badchar[txt[s + j + start]];
			if (a < 1) {
				s += 1;
			}
			else {
				s += a;
			}
		}
	}
}

int fileLen = 0;
char* getFileContents(const char*);
char* contents;
/* Driver code */
int main(int argc, char* argv[]) {
	/*
	input format: <file to search> <search pattern> <CPU | GPU> <num threads>
	*/

	if (argc != 5) {
		cout << "input format: <file to search> <search pattern> <CPU | GPU> <num threads>";
		return -1;
	}
	char* file = argv[1];
	char* pat_ = argv[2];

	int pat_len = strlen(pat_);
	char* pat;
	int* total;

	cudaMallocManaged(&total, sizeof(int));
	cudaMallocManaged(&pat, pat_len);
	for (int i = 0; i < pat_len; i++) {
		pat[i] = pat_[i];
	}

	int numThreads = atoi(argv[4]);

	contents = getFileContents(file);



	int partitionLength = strlen(contents) / numThreads;
	auto start = chrono::high_resolution_clock::now();


	search << <1, numThreads >> > (contents, pat, partitionLength, pat_len,total);
	cudaDeviceSynchronize();


	auto stop = chrono::high_resolution_clock::now();
	auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);

	cout << " Total :" << *total << endl;
	cout << duration.count() << endl;
	cudaFree(contents);
	return 0;
}


char* getFileContents(const char* filename) {
	ifstream in(filename, ios::in | ios::binary);
	if (in) {
		char* contents;
		in.seekg(0, ios::end);
		int len = in.tellg();
		cudaMallocManaged(&contents, len * sizeof(char));
		in.seekg(0, ios::beg);
		in.read(&contents[0], len);
		in.close();
		return(contents);
	}
	throw(errno);
}