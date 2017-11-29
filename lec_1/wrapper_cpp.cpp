#define __CL_ENABLE_EXCEPTIONS
#include "CL/cl.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "profiler.h"
#include <algorithm>

int main(int argc, char** argv)
{
    const size_t n = 1024 * 1024;
    std::vector<double> h_a(n), h_b(n), h_c(n);
    profiler prof;

    std::ifstream ifs("ex_2.cl");
    std::string source( (std::istreambuf_iterator<char>(ifs)), (std::istreambuf_iterator<char>()) );
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    std::mt19937 engine; 
    auto generator = std::bind(distribution, engine);
    std::generate_n(h_a.begin(), n, generator);

    prof.tic("opencl");
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    auto platform_id = platforms[0];
    
    std::vector<cl::Device> devices;
    platform_id.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    auto device_id = devices[0];
    
    cl::Context context(device_id);
    cl::CommandQueue commands(context);
    cl::Program program(context, source, true);

    auto hello = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer>(program, "hello");
    
    cl::Buffer d_a(context, h_a.begin(), h_a.end(), true);
    cl::Buffer d_b(context, h_b.begin(), h_b.end(), true);
    cl::Buffer d_c(context, CL_MEM_WRITE_ONLY, sizeof(double) * n);

    hello( cl::EnqueueArgs(commands, cl::NDRange(n)), d_a, d_b, d_c );

    commands.finish();

    cl::copy(commands, d_c, h_c.begin(), h_c.end());
    prof.toc("opencl");
    
    for(auto i = 0; i < n; ++i)
        if(fabs(h_b[i] + h_a[i] - h_c[i]) > 1e-5)
        {
            std::cout << "Error" << std::endl;
            return 1;
        }

    prof.report();

    return 0;
}
