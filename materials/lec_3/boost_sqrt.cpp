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

    // получаем устройство по умолчанию, создаем контекст и очередь
    compute::device device = compute::system::default_device();
    compute::context context(device);
    compute::command_queue queue(context, device);

    size_t n = 1 << 27;
    std::vector<double> host_vector(n);
    std::vector<double> result_vector(n);
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    std::mt19937 engine; 
    auto generator = std::bind(distribution, engine);
    std::generate_n(host_vector.begin(), n, generator);

    // создаем вектор на устройстве
    compute::vector<double> device_vector(host_vector.size(), context);
    
    prof.tic("boost");
    // копируем данные с хоста на устройство
    compute::copy( host_vector.begin(), host_vector.end(), device_vector.begin(), queue );

    prof.tic("boost kernel");
    // вычисляем квадратный корень поэлементно
    compute::transform( device_vector.begin(), device_vector.end(), device_vector.begin(), compute::sqrt<double>(), queue );
    prof.toc("boost kernel");

    // копируем данные обратно на хост
    compute::copy( device_vector.begin(), device_vector.end(), result_vector.begin(), queue );
    prof.toc("boost");

    prof.tic("std");
    std::transform( host_vector.begin(), host_vector.end(), host_vector.begin(), [](double x) { return sqrt(x);} );
    prof.toc("std");

    prof.report();	

    return 0;
}
