#include <iostream>
#include <vector>
#include <algorithm>
#include "profiler.h"

#include <boost/compute/algorithm/transform.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/functional/math.hpp>

namespace compute = boost::compute;

int main()
{
    profiler prof;

    // get default device and setup context
    compute::device device = compute::system::default_device();
    compute::context context(device);
    compute::command_queue queue(context, device);

    // generate random data on the host
    size_t n = 1024*1024*1024;
    std::vector<double> host_vector(n);
    std::vector<double> result_vector(n);
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    std::mt19937 engine; 
    auto generator = std::bind(distribution, engine);
    std::generate_n(host_vector.begin(), n, generator);

    prof.tic("boost");
    // create a vector on the device
    compute::vector<double> device_vector(host_vector.size(), context);

    // transfer data from the host to the device
    compute::copy( host_vector.begin(), host_vector.end(), device_vector.begin(), queue );

    // calculate the square-root of each element in-place
    compute::transform( device_vector.begin(), device_vector.end(), device_vector.begin(), compute::sqrt<double>(), queue );

    // copy values back to the host
    compute::copy( device_vector.begin(), device_vector.end(), result_vector.begin(), queue );
    prof.toc("boost");

    prof.tic("std");
    std::transform( host_vector.begin(), host_vector.end(), host_vector.begin(), [](double x) { return sqrt(x);} );
    prof.toc("std");

    for(auto i = 0; i < n; ++i)
        if(fabs(host_vector[i] - result_vector[i]) > 1e-5)
        {
            std::cout << "Error" << std::endl;
            return 1;
        }

    prof.report();	

    return 0;
}
