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
    // Open benchmark CSV file to record each run's parameters and runtime.
    std::ofstream bench_file("benchmark.csv");
    if (!bench_file.is_open()) {
        std::cerr << "Error opening benchmark.csv for writing." << std::endl;
        return 1;
    }
    bench_file << "run,num_particles,num_steps,run_time_s\n";

    // Define simulation run parameters.
    int initial_particles = 10;
    int initial_steps = 1000;
    int num_runs = 30;         // Total number of runs (adjust as needed)
    int delta_particles = 10;  // Increase in particle count per run
    int delta_steps = 1000;    // Increase in steps per run

    // Constants (remain unchanged)
    const double kB = 1.38064852e-23;  // Boltzmann constant (J/K)
    const double T = 310;              // Temperature (K)
    const double eta = 3.26e-3;        // Viscosity of blood (N s/m^2)
    const double pi = M_PI;            // Use M_PI from <cmath>
    std::vector<double> time_intervals = {0.1, 0.05, 0.01}; // seconds

    // Limit CPU utilization by setting a fixed number of threads.
    // Change the number (here 4) to use fewer cores if desired.
    // omp_set_num_threads(4);
    omp_set_num_threads(omp_get_max_threads());

    // Loop over simulation runs with increasing parameters.
    for (int run = 0; run < num_runs; run++) {
        // Update simulation parameters for this run.
        int num_particles = initial_particles + run * delta_particles;
        int num_steps = initial_steps + run * delta_steps;

        // Record start time for this run.
        auto run_start = std::chrono::high_resolution_clock::now();

        // Generate global particle sizes for the current number of particles.
        std::vector<double> global_particle_sizes(num_particles, 0.0);
        std::random_device rd;
        {
            std::mt19937 gen(rd());
            std::uniform_real_distribution<double> size_dist(50e-9, 300e-9);
            for (int p = 0; p < num_particles; ++p) {
                global_particle_sizes[p] = size_dist(gen);
            }
        }

        // Open performance CSV file (append mode so that previous content is not overwritten).
        std::ofstream perf_file("performance.csv", std::ios_base::app);
        if (!perf_file.is_open()) {
            std::cerr << "Error opening performance.csv for writing." << std::endl;
            return 1;
        }

        // Loop over each time interval.
        for (double dt : time_intervals) {
            auto combo_start = std::chrono::high_resolution_clock::now();

            // Allocate container for mean square displacement (MSD) values.
            std::vector<std::vector<double>> msd_all(num_particles, std::vector<double>(num_steps, 0.0));

            // Open trajectory file for this dt.
            std::ostringstream traj_filename;
            traj_filename << "traj_dt" << std::fixed << std::setprecision(2) << dt << ".csv";
            std::ofstream traj_file(traj_filename.str());
            if (!traj_file.is_open()) {
                std::cerr << "Error opening file: " << traj_filename.str() << std::endl;
                continue;
            }
            traj_file << "particle,step,time,x,y,z,particle_size\n";

            // Define a flush interval for writing out data.
            const int FLUSH_INTERVAL = 100;

            // Begin parallel region with OpenMP.
            #pragma omp parallel
            {
                // Each thread gets its own random number generator.
                unsigned seed = rd() ^ ((std::mt19937::result_type)omp_get_thread_num() << 1);
                std::mt19937 thread_gen(seed);
                // Use a thread-local ostringstream to buffer output.
                std::ostringstream local_buffer;

                // Parallelize over particles using dynamic scheduling.
                #pragma omp for schedule(dynamic)
                for (int p = 0; p < num_particles; p++) {
                    double particle_size = global_particle_sizes[p];
                    double D = kB * T / (3 * pi * eta * particle_size);
                    double particle_stddev = std::sqrt(2.0 * D * dt);
                    std::normal_distribution<double> particle_dist(0.0, particle_stddev);

                    // Allocate position and MSD vectors for the current particle.
                    std::vector<double> X(num_steps, 0.0);
                    std::vector<double> Y(num_steps, 0.0);
                    std::vector<double> Z(num_steps, 0.0);
                    std::vector<double> local_msd(num_steps, 0.0);

                    // Allocate step displacement vectors.
                    std::vector<double> dx_steps(num_steps, 0.0);
                    std::vector<double> dy_steps(num_steps, 0.0);
                    std::vector<double> dz_steps(num_steps, 0.0);
                    
                    // Generate random displacements.
                    for (int step = 1; step < num_steps; step++) {
                        dx_steps[step] = particle_dist(thread_gen);
                        dy_steps[step] = particle_dist(thread_gen);
                        dz_steps[step] = particle_dist(thread_gen);
                    }
                    // Accumulate positions.
                    for (int step = 1; step < num_steps; step++) {
                        X[step] = X[step - 1] + dx_steps[step];
                        Y[step] = Y[step - 1] + dy_steps[step];
                        Z[step] = Z[step - 1] + dz_steps[step];
                    }
                    // Calculate mean square displacement.
                    #pragma omp simd
                    for (int step = 0; step < num_steps; step++) {
                        local_msd[step] = X[step]*X[step] + Y[step]*Y[step] + Z[step]*Z[step];
                    }
                    msd_all[p] = local_msd;

                    // Write trajectory data.
                    local_buffer << p << "," << 0 << "," << 0.0 << ","
                                 << X[0] << "," << Y[0] << "," << Z[0] << "," << particle_size << "\n";
                    for (int step = 1; step < num_steps; step++) {
                        double time = step * dt;
                        local_buffer << p << "," << step << "," << time << ","
                                     << X[step] << "," << Y[step] << "," << Z[step] << "," << particle_size << "\n";
                        // Flush the local buffer to file periodically.
                        if (step % FLUSH_INTERVAL == 0) {
                            #pragma omp critical
                            {
                                traj_file << local_buffer.str();
                            }
                            local_buffer.str("");  // Clear the local buffer to free memory.
                        }
                    }
                    // Flush any remaining data.
                    #pragma omp critical
                    {
                        traj_file << local_buffer.str();
                    }
                    local_buffer.str("");
                } // end particle loop
            } // end parallel region

            traj_file.close();
            std::cout << "Saved trajectories to " << traj_filename.str() << std::endl;

            // Write MSD data to file.
            std::ostringstream msd_filename;
            msd_filename << "msd_dt" << std::fixed << std::setprecision(2) << dt << ".csv";
            std::ofstream msd_file(msd_filename.str());
            if (!msd_file.is_open()) {
                std::cerr << "Error opening file: " << msd_filename.str() << std::endl;
                continue;
            }
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

            auto combo_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> combo_duration = combo_end - combo_start;
            
            // Record performance metrics for each particle.
            for (int p = 0; p < num_particles; p++) {
                double particle_size = global_particle_sizes[p];
                double D_perf = kB * T / (6 * pi * eta * particle_size);
                double particle_size_nm = particle_size * 1e9;
                perf_file << dt << "," << num_steps << "," << combo_duration.count() << "," 
                          << p << "," << D_perf << "," << particle_size_nm << "\n";
            }
        } // End dt loop

        // Record end time for this simulation run.
        auto run_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> run_duration = run_end - run_start;
        std::cout << "Run " << run << " simulation runtime: " << run_duration.count() << " seconds." << std::endl;
        bench_file << run << "," << num_particles << "," << num_steps << "," << run_duration.count() << "\n";
        perf_file.close();
    } // End run loop

    std::cout << "All simulation runs completed." << std::endl;
    bench_file.close();
    return 0;
}
