#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <random>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <chrono> // For high resolution clock timing

int main() {
    // Start overall simulation timer
    auto start_time = std::chrono::high_resolution_clock::now();

    // Constants
    const double kB = 1.38064852e-23;  // Boltzmann constant (J/K)
    const double T = 310;              // Temperature (K)
    const double eta = 3.26e-3;        // Viscosity of blood (N s/m^2)
    const double pi = M_PI;            // Use the M_PI constant from <cmath>

    // Particle sizes in meters (50 nm, 100 nm, 300 nm)
    std::vector<double> particle_sizes = {50e-9, 100e-9, 300e-9};

    // Calculate diffusion coefficients using the Stokes-Einstein equation:
    // D = kB * T / (6 * pi * eta * particle_size)
    std::vector<double> D_values;
    D_values.reserve(particle_sizes.size());
    for (double size : particle_sizes) {
        double D = kB * T / (6 * pi * eta * size);
        D_values.push_back(D);
        std::cout << "Particle size: " << std::fixed << std::setprecision(1)
                  << size * 1e9 << " nm - Diffusion Coefficient: "
                  << std::scientific << D << " m^2/s" << std::endl;
    }

    // Instead of a single number of particles/steps, define multiple values to test:
    std::vector<int> particle_counts = {10, 50, 100, 300, 500};
    std::vector<int> step_counts     = {500, 1000, 2000, 3500, 5000};

    // Multiple time intervals to test
    std::vector<double> time_intervals = {0.1, 0.05, 0.01}; // seconds

    // Open CSV file for performance metrics.
    // This file will record the simulation parameters and the runtime (time complexity) for each simulation run.
    std::ofstream perf_file("performance.csv");
    if (!perf_file.is_open()) {
        std::cerr << "Error opening performance.csv for writing." << std::endl;
        return 1;
    }
    // CSV header
    perf_file << "time_interval,particle_size_nm,diffusion_coefficient,"
              << "num_particles,num_steps,simulation_time_s\n";

    // Setup random number generation (using a Mersenne Twister engine)
    std::random_device rd;
    std::mt19937 gen(rd());

    // Loop over each time interval
    for (double dt : time_intervals) {
        // For each particle size (and its corresponding diffusion coefficient)
        for (size_t i = 0; i < particle_sizes.size(); ++i) {
            double size = particle_sizes[i];
            double D    = D_values[i];

            // Now loop over each combination of num_particles and num_steps
            for (int num_particles : particle_counts) {
                for (int num_steps : step_counts) {
                    // Record simulation start time for this combination.
                    auto combo_start = std::chrono::high_resolution_clock::now();

                    // Prepare a container for the trajectories.
                    // positions[p][step] holds an array {x, y} for particle 'p' at a given step.
                    std::vector< std::vector< std::array<double, 2> > >
                        positions(num_particles, std::vector<std::array<double, 2>>(num_steps, {0.0, 0.0}));

                    // Standard deviation for the step distribution: sqrt(2 * D * dt)
                    double stddev = std::sqrt(2.0 * D * dt);
                    std::normal_distribution<double> dist(0.0, stddev);

                    // Simulate the trajectories for each particle
                    for (int p = 0; p < num_particles; p++) {
                        // Starting at (0, 0) is already set.
                        for (int step = 1; step < num_steps; step++) {
                            double dx = dist(gen);
                            double dy = dist(gen);
                            positions[p][step][0] = positions[p][step - 1][0] + dx;
                            positions[p][step][1] = positions[p][step - 1][1] + dy;
                        }
                    }

                    // Construct a file name that encodes the simulation parameters for trajectories.
                    // For example: "traj_dt0.10_size50nm_10particles_1000steps.csv"
                    std::ostringstream traj_filename;
                    traj_filename << "traj_dt" << std::fixed << std::setprecision(2) << dt
                                  << "_size" << std::fixed << std::setprecision(0) << size * 1e9 << "nm_"
                                  << num_particles << "particles_"
                                  << num_steps << "steps.csv";
                    std::ofstream traj_file(traj_filename.str());
                    if (!traj_file.is_open()) {
                        std::cerr << "Error opening file: " << traj_filename.str() << std::endl;
                        continue;
                    }

                    // Write CSV header: particle,step,time,x,y
                    traj_file << "particle,step,time,x,y\n";
                    // Write the trajectory data for each particle and each time step.
                    for (int p = 0; p < num_particles; p++) {
                        for (int step = 0; step < num_steps; step++) {
                            double time = step * dt;
                            traj_file << p << "," << step << "," << time << ","
                                      << positions[p][step][0] << ","
                                      << positions[p][step][1] << "\n";
                        }
                    }
                    traj_file.close();
                    std::cout << "Saved trajectories to " << traj_filename.str() << std::endl;

                    // Compute the mean square displacement (MSD) for each time step.
                    std::vector<double> msd(num_steps, 0.0);
                    for (int step = 0; step < num_steps; step++) {
                        double sum_sq = 0.0;
                        for (int p = 0; p < num_particles; p++) {
                            double x = positions[p][step][0];
                            double y = positions[p][step][1];
                            sum_sq += (x * x + y * y);
                        }
                        msd[step] = sum_sq / num_particles;
                    }

                    // Construct a file name for the MSD output.
                    // For example: "msd_dt0.10_size50nm_10particles_1000steps.csv"
                    std::ostringstream msd_filename;
                    msd_filename << "msd_dt" << std::fixed << std::setprecision(2) << dt
                                 << "_size" << std::fixed << std::setprecision(0) << size * 1e9 << "nm_"
                                 << num_particles << "particles_"
                                 << num_steps << "steps.csv";
                    std::ofstream msd_file(msd_filename.str());
                    if (!msd_file.is_open()) {
                        std::cerr << "Error opening file: " << msd_filename.str() << std::endl;
                        continue;
                    }

                    // Write CSV header: step,time,msd
                    msd_file << "step,time,msd\n";
                    for (int step = 0; step < num_steps; step++) {
                        double time = step * dt;
                        msd_file << step << "," << time << "," << msd[step] << "\n";
                    }
                    msd_file.close();
                    std::cout << "Saved MSD to " << msd_filename.str() << std::endl;

                    // Record simulation end time for this combination.
                    auto combo_end = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> combo_duration = combo_end - combo_start;

                    // Write performance metrics to the CSV file.
                    // Format: time_interval, particle_size_nm, diffusion_coefficient, num_particles, num_steps, simulation_time_s
                    perf_file << dt << ","
                              << size * 1e9 << ","
                              << D << ","
                              << num_particles << ","
                              << num_steps << ","
                              << combo_duration.count() << "\n";
                }
            }
        }
    }

    // End overall simulation timer
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "Total simulation runtime: " << elapsed.count() << " seconds." << std::endl;

    perf_file.close();
    return 0;
}
