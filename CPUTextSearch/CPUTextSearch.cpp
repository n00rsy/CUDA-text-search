#include <algorithm>
#include <string>
#include <fstream>
#include <iostream>
#include <thread>
#include <chrono> 

using namespace std;

# define NO_OF_CHARS 256  

long GetFileSize(string);
int total = 0;

// preprocessing function 
void badCharHeuristic(string str, int size,
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
void search(string txt, string pat) {
	int m = pat.size();
	int n = txt.size();

	int badchar[NO_OF_CHARS];

	/* Fill the bad character array by calling
	the preprocessing function badCharHeuristic()
	for given pattern */
	badCharHeuristic(pat, m, badchar);

	int s = 0; // s = shift of pattern
	while (s <= (n - m)) {
		int j = m - 1;
		while (j >= 0 && pat[j] == txt[s + j]) {
			j--;
		}
		if (j < 0) {
			//cout << "found pattern at = " << s << endl;
			s += (s + m < n) ? m - badchar[txt[s + m]] : 1;

			total++;
		}
		else {
			s += max(1, j - badchar[txt[s + j]]);
		}
	}
}

/* Driver code */
int main(int argc, char* argv[]) {
	/*
	input format: <file to search> <search pattern> <CPU | GPU> <num threads>
	*/

	//check all args
	if (argc != 5) {
		cout << "input format: <file to search> <search pattern> <CPU | GPU> <num threads>";
		return -1;
	}
	char* file = argv[1];
	string pat = string(argv[2]);
	int numThreads = atoi(argv[4]);

	ifstream infile;
	infile.open(file);
	thread* threads = new thread[numThreads];
	if (infile.is_open()) {
		auto start = chrono::high_resolution_clock::now();
		//create buffer (size of file/# threads)+ (100 char for possible overflow)
		int bufferLength = GetFileSize(file) / numThreads;
		//cout << "Buffer length: " << bufferLength << endl;
		char* data = (char*)malloc(sizeof(char) * (bufferLength + 100));
		int threadCount = 0;
		while (infile) {

			infile.read(data, bufferLength);
			int i = infile.gcount() - 1;
			while (data[i] != ' ' && i > 0) {
				i--;
				infile.seekg(infile.tellg() - (streampos)1);
			}

			data[i] = '\0';
			//cout << "Searching in:\n " << data << endl;
			//search(string(data), pat);
			threads[threadCount] = thread(search, string(data), pat);
			threadCount++;
		}

		for (int i = 0; i < threadCount; i++) {
			threads[i].join();
		}

		auto stop = chrono::high_resolution_clock::now();
		auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);

		cout << duration.count() << endl;

		infile.close();
		free(data);
	}

	else {
		cout << "Failed to open " << file << endl;
		return -1;
	}
	cout << "Total: " << total << endl;
	return 0;
}

long GetFileSize(string filename)
{
	struct stat stat_buf;
	int rc = stat(filename.c_str(), &stat_buf);
	return rc == 0 ? stat_buf.st_size : -1;
}