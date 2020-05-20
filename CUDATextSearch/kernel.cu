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
__global__ void search(const char* txt, char* pat, int chunk_size, int pat_size, int * t, int threads_per_block) {

	int offset = blockIdx.x* threads_per_block* chunk_size;
	int i = threadIdx.x;
	int start = offset+(i * chunk_size);
	int m = pat_size;
	int n = chunk_size;

	int badchar[NO_OF_CHARS];

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

/* Driver code */
int main(int argc, char* argv[]) {
	/*
	input format: <file to search> <search pattern> <num thread blocks> <num threads>
	*/

	if (argc != 5) {
		cout << "input format: <file to search> <search pattern>  <num threads> <num thread blocks>";
		return -1;
	}
	char* file = argv[1];
	char* pat_ = argv[2];

	int pat_len = strlen(pat_);
	char* pat;
	cudaMallocManaged(&pat, pat_len);
	//copy input pattern from normal memory to shared CPU/GPU memory
	for (int i = 0; i < pat_len; i++) {
		pat[i] = pat_[i];
	}
	//shared count variable - needed by cpu and gpu
	int* total;
	cudaMallocManaged(&total, sizeof(int));

	//should check all these to prevent errors but naa. my project my rules
	char * contents = getFileContents(file);
	int num_threads = atoi(argv[4]);
	int num_blocks = atoi(argv[3]);

	int partitionLength = strlen(contents) / num_threads;
	auto start = chrono::high_resolution_clock::now();

	int threads_per_block = num_threads / num_blocks;
	//make num blocks * threads per block gpu threads
	search <<<num_blocks, threads_per_block >>> (contents, pat, partitionLength, pat_len,total, threads_per_block );
	//wait for GPU operations to terminate
	cudaDeviceSynchronize();

	auto stop = chrono::high_resolution_clock::now();
	auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);

	cout << "Total :" << *total << endl;
	cout << duration.count() << endl;
	//free shared variables
	cudaFree(contents);
	cudaFree(pat);
	cudaFree(total);
	return 0;
}

char* getFileContents(const char* filename) {
	ifstream in(filename, ios::in | ios::binary);
	if (in) {
		in.seekg(0, ios::end);
		int len = in.tellg();

		char* contents;
		cudaMallocManaged(&contents, len * sizeof(char));

		in.seekg(0, ios::beg);
		in.read(&contents[0], len);
		in.close();

		return(contents);
	}
	throw(errno);
}