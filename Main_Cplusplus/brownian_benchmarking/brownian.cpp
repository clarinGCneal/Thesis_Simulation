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
    // Start overall simulation timer
    auto start_time = std::chrono::high_resolution_clock::now();

    // Constants
    const double kB  = 1.38064852e-23;  // Boltzmann constant (J/K)
    const double T   = 310;            // Temperature (K)
    const double eta = 3.26e-3;        // Viscosity of blood (N s/m^2)
    const double pi  = M_PI;           // Use M_PI from <cmath>

    // *** Only ONE particle size: 50 nm in meters ***
    const double particle_size = 50e-9;  

    // Calculate diffusion coefficient (Stokes-Einstein)
    double D = kB * T / (6.0 * pi * eta * particle_size);
    std::cout << std::fixed << std::setprecision(1)
              << "Particle size: " << particle_size * 1e9 << " nm - "
              << "Diffusion Coefficient: " << std::scientific << D
              << " m^2/s\n";

    // Single time interval
    const double dt = 0.1;    
    // Fixed number of steps
    const int num_steps = 5000;

    // Build the list of number of particles: 10 -> 50, then in steps of +50 up to 500
    std::vector<int> numParticlesList;
    for (int n = 10; n <= 1000; /* step decided below */) {
        numParticlesList.push_back(n);
        if (n < 50) {
            // jump from 10 to 50
            n += 40;
        } else {
            // after 50, go in steps of 50
            n += 50;
        }
    }

    // Open CSV file for runtime metrics
    std::ofstream runtime_file("runtime_results.csv");
    if (!runtime_file.is_open()) {
        std::cerr << "Error opening runtime_results.csv for writing." << std::endl;
        return 1;
    }
    // CSV header
    runtime_file << "particle_size_nm,num_particles,num_steps,simulation_time_s\n";

    // Random number generation setup
    std::random_device rd;

    // Standard deviation for each step: sqrt(2 * D * dt)
    double stddev = std::sqrt(2.0 * D * dt);

    // Loop over each number of particles in our list
    for (int num_particles : numParticlesList) {
        // Record start time
        auto combo_start = std::chrono::high_resolution_clock::now();

        // We'll keep an msd vector so the compiler doesn't optimize away the simulation
        std::vector<double> msd(num_steps, 0.0);

        #pragma omp parallel
        {
            // Each thread has a unique seed
            unsigned seed = rd() ^ 
                            (std::mt19937::result_type)omp_get_thread_num() << 1;
            std::mt19937 thread_gen(seed);
            std::normal_distribution<double> thread_dist(0.0, stddev);

            // A local array to accumulate partial msd
            std::vector<double> local_msd(num_steps, 0.0);

            #pragma omp for
            for (int p = 0; p < num_particles; p++) {
                // Positions
                std::vector<double> X(num_steps, 0.0);
                std::vector<double> Y(num_steps, 0.0);

                // Generate random displacements and update positions
                for (int step = 1; step < num_steps; step++) {
                    double dx = thread_dist(thread_gen);
                    double dy = thread_dist(thread_gen);

                    X[step] = X[step - 1] + dx;
                    Y[step] = Y[step - 1] + dy;

                    // Accumulate MSD
                    local_msd[step] += (X[step] * X[step] + Y[step] * Y[step]);
                }
            }

            // Atomically add local_msd into global msd
            #pragma omp critical
            {
                for (int step = 0; step < num_steps; step++) {
                    msd[step] += local_msd[step];
                }
            }
        } // end parallel

        // Measure end time
        auto combo_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> combo_duration = combo_end - combo_start;

        // Write runtime info to CSV
        runtime_file << (particle_size * 1e9) << ","
                     << num_particles       << ","
                     << num_steps          << ","
                     << combo_duration.count() 
                     << "\n";

        // Print to console as well
        std::cout << "Size = " << particle_size * 1e9 << " nm, "
                  << "Particles = " << num_particles << ", "
                  << "Steps = " << num_steps << ", "
                  << "Runtime (s) = " << combo_duration.count() << std::endl;
    }

    // Close the runtime file
    runtime_file.close();

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_elapsed = end_time - start_time;
    std::cout << "Total simulation runtime: " << total_elapsed.count() << " seconds.\n";

    return 0;
}
