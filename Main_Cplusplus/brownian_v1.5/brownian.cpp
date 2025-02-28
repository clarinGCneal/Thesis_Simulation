#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <omp.h>
#ifdef _OPENMP
#endif

int main() {
    // Start overall simulation timer
    auto start_time = std::chrono::high_resolution_clock::now();

    // Constants
    const double kB = 1.38064852e-23;  // Boltzmann constant (J/K)
    const double T = 310;              // Temperature (K)
    const double eta = 3.26e-3;        // Viscosity of blood (N s/m^2)
    const double pi = M_PI;            // Use M_PI from <cmath>

    // Particle sizes in meters (50 nm, 100 nm, 300 nm)
    std::vector<double> particle_sizes = {50e-9, 100e-9, 300e-9};

    // Calculate diffusion coefficients (Stokes-Einstein)
    std::vector<double> D_values;
    D_values.reserve(particle_sizes.size());
    for (double size : particle_sizes) {
        double D = kB * T / (6 * pi * eta * size);
        D_values.push_back(D);
        std::cout << "Particle size: " << std::fixed << std::setprecision(1)
                  << size * 1e9 << " nm - Diffusion Coefficient: "
                  << std::scientific << D << " m^2/s" << std::endl;
    }

    // Simulation parameters
    const int num_particles = 10;        // number of particles per simulation
    const int num_steps = 1000;          // number of time steps per simulation
    std::vector<double> time_intervals = {0.1, 0.05, 0.01}; // seconds

    // Open CSV file for performance metrics
    std::ofstream perf_file("performance.csv");
    if (!perf_file.is_open()) {
        std::cerr << "Error opening performance.csv for writing." << std::endl;
        return 1;
    }
    // CSV header
    perf_file << "time_interval,particle_size_nm,diffusion_coefficient,"
              << "num_particles,num_steps,simulation_time_s\n";

    // Random number generation setup
    std::random_device rd;

    // Loop over each time interval
    for (double dt : time_intervals) {
        // For each particle size (and its corresponding diffusion coefficient)
        for (size_t i = 0; i < particle_sizes.size(); ++i) {
            double size = particle_sizes[i];
            double D = D_values[i];

            // Record simulation start time for this combination
            auto combo_start = std::chrono::high_resolution_clock::now();

            // Standard deviation for each step: sqrt(2 * D * dt)
            double stddev = std::sqrt(2.0 * D * dt);

            // We'll compute MSD in one pass
            std::vector<double> msd(num_steps, 0.0);

            // Trajectory file: open once
            std::ostringstream traj_filename;
            traj_filename << "traj_dt" << std::fixed << std::setprecision(2) << dt
                          << "_size" << std::fixed << std::setprecision(0)
                          << size * 1e9 << "nm.csv";
            std::ofstream traj_file(traj_filename.str());
            if (!traj_file.is_open()) {
                std::cerr << "Error opening file: " << traj_filename.str() << std::endl;
                continue;
            }
            // CSV header
            traj_file << "particle,step,time,x,y\n";

            // Buffer for trajectory lines to reduce I/O overhead
            std::ostringstream buffer;
            const int FLUSH_INTERVAL = 100; // e.g. flush every 100 steps

            // Simulate each particle in parallel
            #pragma omp parallel
            {
                // Each thread has a unique seed
                unsigned seed = rd() ^ (std::mt19937::result_type)omp_get_thread_num() << 1;
                std::mt19937 thread_gen(seed);
                std::normal_distribution<double> thread_dist(0.0, stddev);

                // A local array to accumulate partial msd
                std::vector<double> local_msd(num_steps, 0.0);

                // Thread-local ostringstream for partial trajectory output
                std::ostringstream local_buffer;

                #pragma omp for
                for (int p = 0; p < num_particles; p++) {
                    // Pre-generate random steps for this particle
                    std::vector<double> dx_steps(num_steps, 0.0);
                    std::vector<double> dy_steps(num_steps, 0.0);

                    // Step 0: (x,y) = (0,0) at time=0
                    // We'll store entire trajectory to output correct positions
                    std::vector<double> X(num_steps, 0.0);
                    std::vector<double> Y(num_steps, 0.0);

                    // Generate random displacements
                    // You *can* do #pragma omp simd here for random draws,
                    // but the function call may reduce actual vectorization.
                    for (int step = 1; step < num_steps; step++) {
                        dx_steps[step] = thread_dist(thread_gen);
                        dy_steps[step] = thread_dist(thread_gen);
                    }

                    // Update positions step by step
                    for (int step = 1; step < num_steps; step++) {
                        X[step] = X[step - 1] + dx_steps[step];
                        Y[step] = Y[step - 1] + dy_steps[step];
                        // Accumulate MSD for this step
                        local_msd[step] += (X[step] * X[step] + Y[step] * Y[step]);
                    }

                    // Now write out the trajectory for this particle
                    // step=0
                    local_buffer << p << "," << 0 << "," << 0.0 << "," << X[0] << "," << Y[0] << "\n";

                    // Write steps 1..num_steps-1
                    for (int step = 1; step < num_steps; step++) {
                        double time = step * dt;
                        local_buffer << p << "," << step << "," << time << ","
                                     << X[step] << "," << Y[step] << "\n";

                        // Periodic buffer flush
                        if (step % FLUSH_INTERVAL == 0) {
                            #pragma omp critical
                            {
                                buffer << local_buffer.str();
                                local_buffer.str("");
                            }
                        }
                    }
                } // end for(p)

                // Atomically add local_msd into global msd
                #pragma omp critical
                {
                    for (int step = 0; step < num_steps; step++) {
                        msd[step] += local_msd[step];
                    }
                    // Dump all leftover lines
                    buffer << local_buffer.str();
                }
            } // end parallel

            // Average MSD across particles
            for (int step = 0; step < num_steps; step++) {
                msd[step] /= num_particles;
            }

            // Write leftover buffer lines
            traj_file << buffer.str();
            traj_file.close();
            std::cout << "Saved trajectories to " << traj_filename.str() << std::endl;

            // Save the MSD to a CSV file
            std::ostringstream msd_filename;
            msd_filename << "msd_dt" << std::fixed << std::setprecision(2) << dt
                         << "_size" << std::fixed << std::setprecision(0)
                         << size * 1e9 << "nm.csv";
            std::ofstream msd_file(msd_filename.str());
            if (!msd_file.is_open()) {
                std::cerr << "Error opening file: " << msd_filename.str() << std::endl;
                continue;
            }
            msd_file << "step,time,msd\n";
            for (int step = 0; step < num_steps; step++) {
                double time = step * dt;
                msd_file << step << "," << time << "," << msd[step] << "\n";
            }
            msd_file.close();
            std::cout << "Saved MSD to " << msd_filename.str() << std::endl;

            // Performance measurement
            auto combo_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> combo_duration = combo_end - combo_start;
            perf_file << dt << ","
                      << size * 1e9 << ","
                      << D << ","
                      << num_particles << ","
                      << num_steps << ","
                      << combo_duration.count() << "\n";
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "Total simulation runtime: " << elapsed.count() << " seconds." << std::endl;

    perf_file.close();
    return 0;
}
