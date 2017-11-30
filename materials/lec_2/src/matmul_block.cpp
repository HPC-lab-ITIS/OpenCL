#define __CL_ENABLE_EXCEPTIONS
#include "CL/cl.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include "profiler.h"
#include <algorithm>

int checkError(int err, const char *mes)
{
    if(err != CL_SUCCESS)
    {
       std::cout << mes << std::endl;
       std::cout << err << std::endl;
    }
    return 1;
}

int main(int argc, char** argv)
{
    const size_t n = 2048;
    std::vector<double> h_a(n*n), h_b(n*n), h_c(n*n);
    profiler prof;
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    std::mt19937 engine; 
    auto generator = std::bind(distribution, engine);
    std::generate_n(h_a.begin(), n*n, generator);
    std::generate_n(h_b.begin(), n*n, generator);

    std::ifstream ifs("matmul_block.cl");
    std::string source( (std::istreambuf_iterator<char>(ifs)), (std::istreambuf_iterator<char>()) );
    const char* ProgramSource = source.c_str();
    int err;

    cl_context context;
    cl_command_queue commands;
    cl_program program;
    cl_kernel matmul;
    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_uint num_platforms;
    cl_uint num_devices;

    cl_mem d_a;
    cl_mem d_b;
    cl_mem d_c;

    prof.tic("opencl");
    // Set up platform and GPU device
    err = clGetPlatformIDs(1, &platform_id, &num_platforms);
    checkError(err, "Finding platforms");
    if(num_platforms == 0)
    {
        printf("Found 0 platforms!\n");
        return 1;
    }

    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &num_devices);
    if(device_id == NULL)
        checkError(err, "Finding a device");

    // Create a compute context
    context = clCreateContext(0, 1, &device_id,	NULL, NULL, &err);

    // Create a command queue
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    checkError(err, "Creating command queue");

    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1, (const char **) &ProgramSource, NULL, &err);
    checkError(err, "Creating program");

    // Build the program
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    
    if(err != CL_SUCCESS)
    {
        checkError(err, "Building program");
        return 1;
    }

    // Create the compute kernel from the program
    matmul = clCreateKernel(program, "matmul", &err);
    checkError(err, "Creating kernel");

    d_a  = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(double) * n * n, NULL, &err);
    checkError(err, "Creating buffer d_a");

    d_b  = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(double) * n * n, NULL, &err);
    checkError(err, "Creating buffer d_b");

    d_c  = clCreateBuffer(context,  CL_MEM_READ_WRITE,  sizeof(double) * n * n, NULL, &err);
    checkError(err, "Creating buffer d_c");

    err = clEnqueueWriteBuffer(commands, d_a, CL_TRUE, 0, sizeof(double) * n * n, h_a.data(), 0, NULL, NULL);
    checkError(err, "Copying h_a to device at d_a");

    err = clEnqueueWriteBuffer(commands, d_b, CL_TRUE, 0, sizeof(double) * n * n, h_b.data(), 0, NULL, NULL);
    checkError(err, "Copying h_b to device at d_b");

    err  = clSetKernelArg(matmul, 0, sizeof(cl_mem), &d_a);
    err |= clSetKernelArg(matmul, 1, sizeof(cl_mem), &d_b);
    err |= clSetKernelArg(matmul, 2, sizeof(cl_mem), &d_c);
    err |= clSetKernelArg(matmul, 3, sizeof(int), &n);
    checkError(err, "Setting kernel arguments");

    size_t global[2];
    global[0] = n; global[1] = n;
    size_t local[2];
    local[0] = 32; local[1] = 32;

    err = clEnqueueNDRangeKernel(commands, matmul, 2, NULL, global, local, 0, NULL, NULL);
    checkError(err, "Enqueueing kernel");

    err = clFinish(commands);
    checkError(err, "Waiting for kernel to finish");

    // Read back the results from the compute device
    err = clEnqueueReadBuffer( commands, d_c, CL_TRUE, 0, sizeof(double) * n * n, h_c.data(), 0, NULL, NULL );  
    if (err != CL_SUCCESS)
    {
        checkError(err, "Reading results");
        return 1;
    }

    prof.toc("opencl");

    prof.tic("cpu");
    std::vector<double> r(n * n);
    
#pragma omp parallel for num_threads(32)
    for(int i = 0; i < n; ++i)
        for(int j = 0; j < n; ++j)
        {
            double tmp = 0.;

            for(int k = 0; k < n; ++k)
                tmp += h_a[i * n + k] * h_b[k * n + j];

            r[i * n + j] = fabs(tmp - h_c[i * n + j]);
        }
    prof.toc("cpu");

    std::cout << "Max error: "<< *max_element(r.begin(), r.end()) << std::endl;

    // cleanup then shutdown
    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_c);
    clReleaseProgram(program);
    clReleaseKernel(matmul);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    prof.report();

    return 0;
}
