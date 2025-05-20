#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <numbers>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <omp.h>

// Function to explicitly reset OpenMP environment before each parallel region
void resetOpenMPEnvironment(int num_desired_threads) {
    // Force reset of OpenMP state
    omp_set_nested(0);
    omp_set_dynamic(0);
    omp_set_num_threads(num_desired_threads);
    
    // Print confirmation of OpenMP settings
    std::cout << "OpenMP environment: threads=" << num_desired_threads 
              << ", dynamic=" << omp_get_dynamic() 
              << ", nested=" << omp_get_nested() << std::endl;
}

int main() {
    // Initialize OpenMP environment explicitly at the start
    omp_set_nested(0);
    omp_set_dynamic(0);
    
    // Define simulation run parameters.
    int initial_particles = 10;
    int initial_steps = 1000;
    int num_runs = 1;         // Total number of runs (adjust as needed)
    int delta_particles = 10;  // Increase in particle count per run
    int delta_steps = 1000;    // Increase in steps per run

    // Constants (remain unchanged)
    const double kB = 1.38064852e-23;               // Boltzmann constant (J/K)
    const double T = 310;                           // Temperature (K)
    const double eta = 3.26e-3;                     // Viscosity of blood (N s/m^2)
    const double pi = std::numbers::pi;             // Pi constant from <numbers> header
    std::vector<double> time_intervals = {0.1};     // seconds

    // Open benchmark CSV file to record each run's parameters and runtime.
    std::ofstream bench_file("benchmark.csv");
    if (!bench_file.is_open()) {
        std::cerr << "Error opening benchmark.csv for writing." << std::endl;
        return 1;
    }
    bench_file << "run,num_particles,num_steps,run_time_s\n";

    // Write header for performance.csv before any simulation runs.
    {
        std::ofstream perf_header("performance.csv");
        if (!perf_header.is_open()) {
            std::cerr << "Error opening performance.csv for writing header." << std::endl;
            return 1;
        }
        // Write header columns.
        perf_header << "time_interval,num_steps,simulation_time_s,particle_index,diffusion_coefficient,particle_size_nm\n";
        // The file is automatically closed at the end of this scope.
    }

    // Cache line size for avoiding false sharing (typical value is 64 bytes)
    const int CACHE_LINE_SIZE = 64;
    const int PADDING = CACHE_LINE_SIZE / sizeof(double);

    // Limit CPU utilization by setting a fixed number of threads.
    const int num_threads = omp_get_max_threads();
    resetOpenMPEnvironment(num_threads);

    // Pre-allocate memory for all runs to reduce reallocations
    int max_particles = initial_particles + (num_runs - 1) * delta_particles;
    int max_steps = initial_steps + (num_runs - 1) * delta_steps;

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
            // This is from 50 nm to 300 nm, presumably the diameter.
            for (int p = 0; p < num_particles; ++p) {
                global_particle_sizes[p] = size_dist(gen);
            }
        }

        // Pre-allocate thread-local random generators to avoid contention
        std::vector<std::mt19937> thread_gens(num_threads);
        for (int i = 0; i < num_threads; i++) {
            thread_gens[i].seed(rd() ^ ((std::mt19937::result_type)i << 1));
        }

        // Loop over each time interval.
        for (double dt : time_intervals) {
            // Open performance CSV file in append mode.
            std::ofstream perf_file("performance.csv", std::ios_base::app);
            if (!perf_file.is_open()) {
                std::cerr << "Error opening performance.csv for appending." << std::endl;
                return 1;
            }

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

            // Create thread-local buffers that we'll collect later
            std::vector<std::string> particle_buffers(num_particles);

            // Calculate an estimate of string size for pre-allocation
            // Each line is ~50-70 chars: p,step,time,x,y,z,size\n
            const size_t estimated_line_size = 70;
            const size_t buffer_size_estimate = num_steps * estimated_line_size;
            
            // Reset OpenMP environment before parallel region to ensure consistency
            resetOpenMPEnvironment(num_threads);
            
            // Begin parallel region with OpenMP.
            #pragma omp parallel
            {
                const int thread_id = omp_get_thread_num();
                std::mt19937& thread_gen = thread_gens[thread_id];
                std::ostringstream local_buffer;
                // Pre-allocate the string that backs the ostringstream
                std::string temp_str;
                temp_str.reserve(buffer_size_estimate);
                local_buffer.str(temp_str);

                // Parallelize over particles using guided scheduling for better load balancing
                #pragma omp for schedule(guided)
                for (int p = 0; p < num_particles; p++) {
                    double particle_size = global_particle_sizes[p];
                    double D = kB * T / (3 * pi * eta * particle_size);
                    double particle_stddev = std::sqrt(2.0 * D * dt);
                    std::normal_distribution<double> particle_dist(0.0, particle_stddev);

                    // Align vectors for better SIMD performance
                    alignas(32) std::vector<double> X(num_steps, 0.0);
                    alignas(32) std::vector<double> Y(num_steps, 0.0);
                    alignas(32) std::vector<double> Z(num_steps, 0.0);
                    alignas(32) std::vector<double> local_msd(num_steps, 0.0);

                    alignas(32) std::vector<double> dx_steps(num_steps, 0.0);
                    alignas(32) std::vector<double> dy_steps(num_steps, 0.0);
                    alignas(32) std::vector<double> dz_steps(num_steps, 0.0);
                    
                    // Generate all random displacements at once
                    #pragma omp simd
                    for (int step = 1; step < num_steps; step++) {
                        dx_steps[step] = particle_dist(thread_gen);
                        dy_steps[step] = particle_dist(thread_gen);
                        dz_steps[step] = particle_dist(thread_gen);
                    }
                    
                    // Calculate positions with explicit vectorization
                    #pragma omp simd
                    for (int step = 1; step < num_steps; step++) {
                        X[step] = X[step - 1] + dx_steps[step];
                        Y[step] = Y[step - 1] + dy_steps[step];
                        Z[step] = Z[step - 1] + dz_steps[step];
                    }
                    
                    // Calculate MSD with explicit vectorization hint
                    #pragma omp simd
                    for (int step = 0; step < num_steps; step++) {
                        local_msd[step] = X[step]*X[step] + Y[step]*Y[step] + Z[step]*Z[step];
                    }
                    msd_all[p] = local_msd;

                    // Build trajectory data in memory buffer
                    local_buffer << p << "," << 0 << "," << 0.0 << ","
                                << X[0] << "," << Y[0] << "," << Z[0] << "," << particle_size << "\n";
                    for (int step = 1; step < num_steps; step++) {
                        double time = step * dt;
                        local_buffer << p << "," << step << "," << time << ","
                                    << X[step] << "," << Y[step] << "," << Z[step] << "," << particle_size << "\n";
                    }
                    
                    // Store in buffer array instead of writing to disk within the parallel region
                    particle_buffers[p] = local_buffer.str();
                    local_buffer.str("");
                } // end particle loop
            } // end parallel region

            // Write all trajectory data at once outside the parallel region
            for (int p = 0; p < num_particles; p++) {
                traj_file << particle_buffers[p];
            }

            traj_file.close();
            std::cout << "Saved trajectories to " << traj_filename.str() << std::endl;

            // Similarly, buffer MSD data before writing
            std::string msd_buffer;
            // Pre-allocate buffer for MSD data
            msd_buffer.reserve(num_particles * num_steps * estimated_line_size);
            
            msd_buffer = "particle,step,time,msd,particle_size\n";
            std::ostringstream msd_line;
            
            for (int p = 0; p < num_particles; p++) {
                for (int step = 0; step < num_steps; step++) {
                    double time = step * dt;
                    msd_line.str("");  // Clear the stream
                    msd_line << p << "," << step << "," << time << ","
                            << msd_all[p][step] << "," << global_particle_sizes[p] << "\n";
                    msd_buffer += msd_line.str();
                }
            }
            
            std::ostringstream msd_filename;
            msd_filename << "msd_dt" << std::fixed << std::setprecision(2) << dt << ".csv";
            std::ofstream msd_file(msd_filename.str());
            if (!msd_file.is_open()) {
                std::cerr << "Error opening file: " << msd_filename.str() << std::endl;
                continue;
            }
            
            msd_file << msd_buffer;
            msd_file.close();
            std::cout << "Saved MSD to " << msd_filename.str() << std::endl;

            auto combo_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> combo_duration = combo_end - combo_start;
            
            // Buffer performance metrics before writing
            std::string perf_buffer;
            // Pre-allocate buffer for performance data
            perf_buffer.reserve(num_particles * estimated_line_size);
            
            std::ostringstream perf_line;
            
            // Record performance metrics for each particle.
            for (int p = 0; p < num_particles; p++) {
                double particle_size = global_particle_sizes[p];
                double D_perf = kB * T / (3 * pi * eta * particle_size);
                double particle_size_nm = particle_size * 1e9;
                
                perf_line.str("");  // Clear the stream
                perf_line << dt << "," << num_steps << "," << combo_duration.count() << "," 
                        << p << "," << D_perf << "," << particle_size_nm << "\n";
                perf_buffer += perf_line.str();
            }
            
            perf_file << perf_buffer;
            perf_file.close();
        } // End dt loop

        auto run_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> run_duration = run_end - run_start;
        std::cout << "Run " << run << " simulation runtime: " << run_duration.count() << " seconds." << std::endl;
        bench_file << run << "," << num_particles << "," << num_steps << "," << run_duration.count() << "\n";
    } // End run loop

    std::cout << "All simulation runs completed." << std::endl;
    bench_file.close();
    return 0;
}