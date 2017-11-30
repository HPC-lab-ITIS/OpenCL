#include <iostream>
#include <vector>
#include <algorithm>
#include "profiler.h"

#include <boost/compute.hpp>

namespace compute = boost::compute;
using compute::lambda::_1;

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
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    std::mt19937 engine; 
    auto generator = std::bind(distribution, engine);
    std::generate_n(host_vector.begin(), n, generator);

    // create a vector on the device
    compute::vector<double> device_vector(host_vector.size(), context);

    prof.tic("h2d");
    // transfer data from the host to the device
    compute::copy( host_vector.begin(), host_vector.end(), device_vector.begin(), queue );
    queue.finish();
    prof.toc("h2d");

    prof.tic("1st");
    //1
    compute::function<double (double)> mul_2 =
        compute::make_function_from_source<double (double)>(
                "mul_2",
                "double mul_2(double x) { return 2. * x; }"
                );
    compute::transform(device_vector.begin(), device_vector.end(), device_vector.begin(), mul_2, queue);
    queue.finish();
    prof.toc("1st");

    prof.tic("2nd");
    //2
    BOOST_COMPUTE_FUNCTION(double, mul_2_v2, (double x),
            {
            return 2. * x;
            });
    compute::transform(device_vector.begin(), device_vector.end(), device_vector.begin(), mul_2_v2, queue);
    queue.finish();
    prof.toc("2nd");

    prof.tic("3rd");
    //3
    compute::transform(device_vector.begin(), device_vector.end(), device_vector.begin(), _1 * 2., queue);
    queue.finish();
    prof.toc("3rd");

    prof.tic("4th");
    //4
    compute::function<double(double)> mul_2_v3 = 2. * _1;
    compute::transform(device_vector.begin(), device_vector.end(), device_vector.begin(), mul_2_v3, queue);
    queue.finish();
    prof.toc("4th");

    // copy values back to the host
    prof.tic("d2h");
    compute::copy( device_vector.begin(), device_vector.end(), host_vector.begin(), queue );
    queue.finish();
    prof.toc("d2h");

    prof.report();

    return 0;
}
