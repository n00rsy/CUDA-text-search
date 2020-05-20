#include <algorithm>
#include <string>
#include <fstream>
#include <iostream>
#include <thread>
#include <chrono> 
#include <string.h>

using namespace std;
# define NO_OF_CHARS 256  

int total = 0;
char* getFileContents(const char*);
// preprocessing function 
void badCharHeuristic(char * str, int size,
	int badchar[NO_OF_CHARS]) {
	int i;

	// Initialize all occurrences as -1  
	for (i = 0; i < NO_OF_CHARS; i++)
		badchar[i] = -1;
	// put the actual value of last occurrence  
	for (i = 0; i < size; i++)
		badchar[(int)str[i]] = i;
}

//implementation Boyer Moore Algorithm
void search(char * txt, char * pat,int chunk_size,int pat_size, int threadCount) {
	int m = pat_size; //pattern size
	int n = chunk_size; //chunk size
	int offset = threadCount * chunk_size;
	int badchar[NO_OF_CHARS];

	//fill bar char array
	badCharHeuristic(pat, m, badchar);
	//cout << pat << endl;
	int s = 0; // s = shift of pattern
	while (s <= (n - m)) {
		int j = m - 1;
		while (j >= 0 && pat[j] == txt[s + j+offset]) {
			j--;
		}
		if (j < 0) {
			//cout << "found pattern at = " << s << endl;
			s += (s + m < n) ? m - badchar[txt[s + m+ offset]] : 1;

			total++;
		}
		else {
			s += max(1, j - badchar[txt[s + j+ offset]]);
		}
	}
}

/* Driver code */
int main(int argc, char* argv[]) {
	/*
	input format: <file to search> <search pattern> <num threads>
	*/

	//check all args
	if (argc != 4) {
		cout << "input format: <file to search> <search pattern> <num threads>";
		return -1;
	}
	char* file = argv[1];
	char * pat = argv[2];
	int numThreads = atoi(argv[3]);

	thread* threads = new thread[numThreads];

	ifstream infile;
	infile.open(file);
	if (infile.is_open()) {
		char* contents = getFileContents(file);

		int thread_count = 0;
		int chunk_size = strlen(contents) / numThreads;
		int pat_size = strlen(pat);

		auto start = chrono::high_resolution_clock::now();
		for (int i = 0; i < numThreads; i++) {
			threads[i] = thread(search, contents, pat, chunk_size, pat_size, i);

			thread_count++;
			//cout << i << endl;
			//search(contents, pat, chunk_size, pat_size, i);
		}

		for (int i = 0; i < thread_count; i++) {
			threads[i].join();
		}

		auto stop = chrono::high_resolution_clock::now();
		auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);

		cout << duration.count() << endl;

		infile.close();
		free(contents);
	}

	else {
		cout << "Failed to open " << file << endl;
		return -1;
	}
	cout << "Total: " << total << endl;
	return 0;
}

char* getFileContents(const char* filename) {
	ifstream in(filename, ios::in | ios::binary);
	if (in) {
		in.seekg(0, ios::end);
		int len = in.tellg();
		char* contents = (char *)malloc( len * sizeof(char));
		in.seekg(0, ios::beg);
		in.read(&contents[0], len);
		in.close();
		return(contents);
	}
	throw(errno);
}