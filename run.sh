#!/bin/bash

export OMP_NUM_THREADS=4
g++ new.cpp -fopenmp -o main
./main
rm main
