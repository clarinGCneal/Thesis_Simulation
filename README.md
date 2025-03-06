# Thesis_Simulation

This is my repo for the Optimizing Brownian Motion Simulation in *C++*.

Using Ubuntu and WSL in Windows Platform

---

# Commands for Compiling
- Normal C++ Compile: `g++ demo.cpp -o demo`

## OpenCilk Commands
- Fibonnaci in C: `/opt/opencilk/bin/clang -fopencilk -O3 fib.c -o fib`
- test_cilk in C++: `clang++ -fopencilk -o test_cilk test_cilk.cpp`
- Execute command under same directory: `./fib`

## OpenMP Commands
- test_openmp in C++: `g++ -fopenmp -o test_openmp test_openmp.cpp`

## Mounting Ubuntu to D: drive
- use `cd /mnt/d` to go to `D:` drive

## Project Compile Commands (C++)
- `g++ -std=c++11 -o brownianWmeasurements brownianWmeasurements.cpp`
    - Name of the output compiled file `brownianWmeasurement`
    - Name of the file you want to compile `brownianWmeasurements.cpp` 

## Compile with optimization flags (Compile in v.1.6).
- `g++ -O3 -fopenmp -march=native brownian.cpp -o brownian_simulation`
`
