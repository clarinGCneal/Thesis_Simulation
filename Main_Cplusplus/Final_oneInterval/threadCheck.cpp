#include <iostream>
#include <omp.h>

int main() {
    // Get the maximum number of threads available
    int max_threads = omp_get_max_threads();
    std::cout << "Maximum number of threads available: " << max_threads << std::endl;
    
    // Get the number of processors/cores
    int num_procs = omp_get_num_procs();
    std::cout << "Number of processors available: " << num_procs << std::endl;
    
    // Demonstrate thread usage with a parallel region
    std::cout << "\nDemonstrating thread usage:\n";
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        
        // Use critical section to prevent output lines from getting mixed up
        #pragma omp critical
        {
            std::cout << "Thread " << thread_id << " of " << omp_get_num_threads() << " is running." << std::endl;
        }
    }
    
    return 0;
}