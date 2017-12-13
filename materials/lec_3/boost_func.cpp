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

    compute::device device = compute::system::default_device();
    compute::context context(device);
    compute::command_queue queue(context, device);

    size_t n = 1 << 27;
    std::vector<double> host_vector(n);
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    std::mt19937 engine; 
    auto generator = std::bind(distribution, engine);
    std::generate_n(host_vector.begin(), n, generator);

    compute::vector<double> device_vector(host_vector.size(), context);

    prof.tic("host to device");
    compute::copy( host_vector.begin(), host_vector.end(), device_vector.begin(), queue );
    queue.finish();
    prof.toc("host to device");

    prof.tic("boost function from source");
    compute::function<double (double)> sin_func =
        compute::make_function_from_source<double (double)>(
                "sin_func",
                "double sin_func(double x) { return sin(x); }"
                );
    compute::transform(device_vector.begin(), device_vector.end(), device_vector.begin(), sin_func, queue);
    queue.finish();
    prof.toc("boost function from source");

    prof.tic("boost function object");
    BOOST_COMPUTE_FUNCTION(double, sin_func_v2, (double x),
            {
            return sin(x);
            });
    compute::transform(device_vector.begin(), device_vector.end(), device_vector.begin(), sin_func_v2, queue);
    queue.finish();
    prof.toc("boost function object");

    prof.tic("boost placeholders");
    compute::transform(device_vector.begin(), device_vector.end(), device_vector.begin(), sin(_1), queue);
    queue.finish();
    prof.toc("boost placeholders");

    prof.tic("boost function w/placeholders");
    compute::function<double(double)> sin_func_v3 = sin(_1);
    compute::transform(device_vector.begin(), device_vector.end(), device_vector.begin(), sin_func_v3, queue);
    queue.finish();
    prof.toc("boost function w/placeholders");

    prof.tic("device to host");
    compute::copy( device_vector.begin(), device_vector.end(), host_vector.begin(), queue );
    queue.finish();
    prof.toc("device to host");

    prof.report();

    return 0;
}
