// TestDriver.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <iostream> 
#include <string>
#include <filesystem>
#include <windows.h>

using namespace std;

int main(){

    //args = file path, pattern, threads
    string CPU_COMMAND = "CPUTextSearch.exe ";
    //args = file path, pattern, thread blocks, threads
    string GPU_COMMAND = "CUDATextSearch.exe ";
    //pattern to search for
    string pattern = "thou";
    //number of times to repeat each search
    int NUM_REPITITIONS = 50;
    //path to test files
    const char* test_path = "..\\test_files\\";
    
    string s = "echo Threads: ";
    string cpu_out = " >> cpuout.txt";
    string gpu_out = " >> gpuout.txt";

    //num threads increases by powers of 2
    for (int i = 1; i <= 64; i *= 2) {
        //iterate through test files
        system((s + to_string(i)+cpu_out).c_str());
        for (const auto& entry : std::filesystem::directory_iterator::directory_iterator(test_path)) {
            //run each test sequence 100 times
            for (int j = 0; j < NUM_REPITITIONS; j++) {
               string cmd = CPU_COMMAND + entry.path().string() + " " + pattern + " " + to_string(i) + cpu_out;
               cout << cmd << endl;
               system(cmd.c_str());
               cout << endl;
            }
        }
    }

    //num threads increases by powers of 2
    for (int i = 256; i <= 4096; i *= 2) {
        system((s + to_string(i) + gpu_out).c_str());
        //iterate through block count
        for (int j = 1; j <= 32; j *= 2) {
            //iterate through test files
            for (const auto& entry : std::filesystem::directory_iterator::directory_iterator(test_path)) {
                //run each test 100 times
                for (int k = 0; k < NUM_REPITITIONS; k++) {
                     string cmd = GPU_COMMAND + entry.path().string() + " " + pattern + " "+to_string(j)+" " + to_string(i)+ gpu_out;
                     cout << cmd << endl;
                     system(cmd.c_str());
                     cout << endl;
                }
            }
        }
    }
    system("PAUSE");
}