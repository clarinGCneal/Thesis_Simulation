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

    // Simulation parameters
    const int num_particles = 10;        // number of particles per simulation
    const int num_steps = 1000;          // number of time steps per simulation
    std::vector<double> time_intervals = {0.1, 0.05, 0.01}; // seconds

    // Define realistic blood particle size range (in meters)
    const double min_particle_size = 50e-9;   // 50 nm
    const double max_particle_size = 300e-9;    // 300 nm

    // Random device for seeding
    std::random_device rd;

    // Generate global particle sizes once for all dt values
    std::vector<double> global_particle_sizes(num_particles, 0.0);
    {
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> size_dist(min_particle_size, max_particle_size);
        for (int p = 0; p < num_particles; ++p) {
            global_particle_sizes[p] = size_dist(gen);
        }
    }

    // Open CSV file for performance metrics with additional columns
    std::ofstream perf_file("performance.csv");
    if (!perf_file.is_open()) {
        std::cerr << "Error opening performance.csv for writing." << std::endl;
        return 1;
    }
    // CSV header: time_interval,num_particles,num_steps,simulation_time_s,particle,diffusion_coefficient,particle_size_nm
    perf_file << "time_interval,num_particles,num_steps,simulation_time_s,particle,diffusion_coefficient,particle_size_nm\n";

    // Loop over each time interval
    for (double dt : time_intervals) {
        // Record simulation start time for this dt
        auto combo_start = std::chrono::high_resolution_clock::now();

        // Containers to store per-particle MSD
        std::vector<std::vector<double>> msd_all(num_particles, std::vector<double>(num_steps, 0.0));
        
        // Trajectory file: naming depends on dt
        std::ostringstream traj_filename;
        traj_filename << "traj_dt" << std::fixed << std::setprecision(2) << dt << ".csv";
        std::ofstream traj_file(traj_filename.str());
        if (!traj_file.is_open()) {
            std::cerr << "Error opening file: " << traj_filename.str() << std::endl;
            continue;
        }
        // CSV header now includes particle_size column (in meters)
        traj_file << "particle,step,time,x,y,z,particle_size\n";

        // Buffer for trajectory lines to reduce I/O overhead
        std::ostringstream buffer;
        const int FLUSH_INTERVAL = 100; // flush every 100 steps

        // Simulate each particle in parallel
        #pragma omp parallel
        {
            // Each thread gets a unique seed
            unsigned seed = rd() ^ ((std::mt19937::result_type)omp_get_thread_num() << 1);
            std::mt19937 thread_gen(seed);

            std::ostringstream local_buffer;

            #pragma omp for
            for (int p = 0; p < num_particles; p++) {
                // Use the global particle size instead of generating a new one
                double particle_size = global_particle_sizes[p];

                // Calculate the diffusion coefficient based on this size for trajectory
                double D = kB * T / (3 * pi * eta * particle_size);
                // Standard deviation for displacement distribution: sqrt(2 * D * dt)
                double particle_stddev = std::sqrt(2.0 * D * dt);
                // Create a normal distribution for the particle's displacement
                std::normal_distribution<double> particle_dist(0.0, particle_stddev);

                // Vectors for trajectory: starting at (0,0,0)
                std::vector<double> X(num_steps, 0.0);
                std::vector<double> Y(num_steps, 0.0);
                std::vector<double> Z(num_steps, 0.0);
                // Vector for storing MSD for this particle
                std::vector<double> local_msd(num_steps, 0.0);

                // Pre-generate random steps for steps 1 to num_steps-1
                std::vector<double> dx_steps(num_steps, 0.0);
                std::vector<double> dy_steps(num_steps, 0.0);
                std::vector<double> dz_steps(num_steps, 0.0);
                for (int step = 1; step < num_steps; step++) {
                    dx_steps[step] = particle_dist(thread_gen);
                    dy_steps[step] = particle_dist(thread_gen);
                    dz_steps[step] = particle_dist(thread_gen);
                }

                // Cumulative sum: update positions sequentially due to dependency
                for (int step = 1; step < num_steps; step++) {
                    X[step] = X[step - 1] + dx_steps[step];
                    Y[step] = Y[step - 1] + dy_steps[step];
                    Z[step] = Z[step - 1] + dz_steps[step];
                }

                // Vectorized MSD computation using OpenMP simd (each iteration is independent)
                #pragma omp simd
                for (int step = 0; step < num_steps; step++) {
                    local_msd[step] = X[step] * X[step] + Y[step] * Y[step] + Z[step] * Z[step];
                }

                // Store the computed MSD for this particle
                msd_all[p] = local_msd;

                // Write out the trajectory for this particle, including its fixed particle size (in meters)
                local_buffer << p << "," << 0 << "," << 0.0 << ","
                             << X[0] << "," << Y[0] << "," << Z[0] << "," << particle_size << "\n";
                for (int step = 1; step < num_steps; step++) {
                    double time = step * dt;
                    local_buffer << p << "," << step << "," << time << ","
                                 << X[step] << "," << Y[step] << "," << Z[step] << "," << particle_size << "\n";
                    if (step % FLUSH_INTERVAL == 0) {
                        #pragma omp critical
                        {
                            buffer << local_buffer.str();
                            local_buffer.str("");
                        }
                    }
                }
                // Flush any remaining data from the local buffer
                #pragma omp critical
                {
                    buffer << local_buffer.str();
                }
            } // end for each particle
        } // end parallel region

        // Write buffered trajectory data to file
        traj_file << buffer.str();
        traj_file.close();
        std::cout << "Saved trajectories to " << traj_filename.str() << std::endl;

        // Save the per-particle MSD to a CSV file with particle and particle_size columns
        std::ostringstream msd_filename;
        msd_filename << "msd_dt" << std::fixed << std::setprecision(2) << dt << ".csv";
        std::ofstream msd_file(msd_filename.str());
        if (!msd_file.is_open()) {
            std::cerr << "Error opening file: " << msd_filename.str() << std::endl;
            continue;
        }
        // CSV header: particle, step, time, msd, particle_size
        msd_file << "particle,step,time,msd,particle_size\n";
        for (int p = 0; p < num_particles; p++) {
            for (int step = 0; step < num_steps; step++) {
                double time = step * dt;
                msd_file << p << "," << step << "," << time << ","
                         << msd_all[p][step] << "," << global_particle_sizes[p] << "\n";
            }
        }
        msd_file.close();
        std::cout << "Saved MSD to " << msd_filename.str() << std::endl;

        // Performance measurement for this dt
        auto combo_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> combo_duration = combo_end - combo_start;
        
        // For each particle, calculate diffusion coefficient and convert particle size to nm,
        // then write the performance and particle-specific info to the CSV file.
        for (int p = 0; p < num_particles; p++) {
            double particle_size = global_particle_sizes[p];
            // Compute the diffusion coefficient for this particle (using a different constant factor)
            double D_perf = kB * T / (6 * pi * eta * particle_size);
            // Convert particle size from meters to nanometers
            double particle_size_nm = particle_size * 1e9;
            perf_file << dt << "," << num_particles << "," << num_steps << ","
                      << combo_duration.count() << "," << p << "," << D_perf << "," << particle_size_nm << "\n";
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "Total simulation runtime: " << elapsed.count() << " seconds." << std::endl;
    perf_file.close();
    return 0;
}
