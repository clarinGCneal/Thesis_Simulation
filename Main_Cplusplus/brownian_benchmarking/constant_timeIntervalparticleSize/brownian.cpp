#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <omp.h>

int main() {
    // Start overall timer
    auto start_time = std::chrono::high_resolution_clock::now();

    // Constants
    const double kB  = 1.38064852e-23;  // Boltzmann constant (J/K)
    const double T   = 310;            // Temperature (K)
    const double eta = 3.26e-3;        // Viscosity of blood (N s/m^2)
    const double pi  = M_PI;           // Use M_PI from <cmath>

    // We use only one particle size: 50 nm
    const double particle_size = 50e-9;
    double D = kB * T / (6.0 * pi * eta * particle_size);

    std::cout << std::fixed << std::setprecision(1)
              << "Particle size: " << (particle_size * 1e9) << " nm - "
              << "Diff. Coefficient: " << std::scientific << D << " m^2/s\n";

    // Single time interval
    const double dt = 0.1;

    // We’ll vary the number of particles and number of steps:
    std::vector<int> nList = {10, 100, 1000, 2000, 5000};
    std::vector<int> stepList = {1000, 5000, 10000, 20000};

    // Open CSV
    std::ofstream runtime_file("runtime_results.csv");
    if (!runtime_file.is_open()) {
        std::cerr << "Error opening runtime_results.csv for writing.\n";
        return 1;
    }
    runtime_file << "particle_size_nm,num_particles,num_steps,simulation_time_s\n";

    // Random number setup
    std::random_device rd;
    double stddev = std::sqrt(2.0 * D * dt);

    // Loop over each combination of (num_particles, num_steps)
    for (int num_particles : nList) {
        for (int num_steps : stepList) {
            // Begin timing
            auto combo_start = std::chrono::high_resolution_clock::now();

            // The main simulation: We'll allocate an MSD array for the steps
            std::vector<double> msd(num_steps, 0.0);

            #pragma omp parallel
            {
                unsigned seed = rd() ^
                                (std::mt19937::result_type)omp_get_thread_num() << 1;
                std::mt19937 thread_gen(seed);
                std::normal_distribution<double> thread_dist(0.0, stddev);

                // Thread‐local partial MSD
                std::vector<double> local_msd(num_steps, 0.0);

                // Loop over all particles
                #pragma omp for
                for (int p = 0; p < num_particles; p++) {
                    // Positions for each step
                    std::vector<double> X(num_steps, 0.0);
                    std::vector<double> Y(num_steps, 0.0);

                    // Sequential random‐walk updates (cannot be vectorized due to dependency)
                    for (int step = 1; step < num_steps; step++) {
                        double dx = thread_dist(thread_gen);
                        double dy = thread_dist(thread_gen);

                        X[step] = X[step - 1] + dx;
                        Y[step] = Y[step - 1] + dy;
                    }

                    // Vectorized accumulation of squared displacement using SIMD
                    #pragma omp simd
                    for (int step = 1; step < num_steps; step++) {
                        local_msd[step] += X[step] * X[step] + Y[step] * Y[step];
                    }
                }
            } // end parallel

            // End timing
            auto combo_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> combo_duration = combo_end - combo_start;

            // Write runtime info to CSV
            runtime_file << (particle_size * 1e9) << ","
                         << num_particles       << ","
                         << num_steps          << ","
                         << combo_duration.count() 
                         << "\n";

            // Print to console
            std::cout << "Particles=" << num_particles 
                      << ", Steps="   << num_steps
                      << ", Runtime=" << combo_duration.count() << " s\n";
        }
    }

    runtime_file.close();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_elapsed = std::chrono::duration<double>(end_time - start_time).count();
    std::cout << "Total time for all runs: " << total_elapsed << " seconds.\n";

    return 0;
}
