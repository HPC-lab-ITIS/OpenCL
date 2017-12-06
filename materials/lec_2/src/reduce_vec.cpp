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
    const size_t n = 1 << 28;
    const size_t global = n / 256;
    const size_t local = 1024;

    std::vector<float> h_a(n), h_result(global / local);
    profiler prof;
    std::uniform_real_distribution<float> distribution(0.0, 1.0);
    std::mt19937 engine; 
    auto generator = std::bind(distribution, engine);
    std::generate_n(h_a.begin(), n, generator);

    std::ifstream ifs("reduce_vec.cl");
    std::string source( (std::istreambuf_iterator<char>(ifs)), (std::istreambuf_iterator<char>()) );
    const char* ProgramSource = source.c_str();
    int err;

    cl_context context;
    cl_command_queue commands;
    cl_program program;
    cl_kernel reduce;
    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_uint num_platforms;
    cl_uint num_devices;

    cl_mem d_a;
    cl_mem d_result;

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

    context = clCreateContext(0, 1, &device_id,	NULL, NULL, &err);

    commands = clCreateCommandQueue(context, device_id, 0, &err);
    checkError(err, "Creating command queue");

    program = clCreateProgramWithSource(context, 1, (const char **) &ProgramSource, NULL, &err);
    checkError(err, "Creating program");

    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    
    if(err != CL_SUCCESS)
    {
        checkError(err, "Building program");
        return 1;
    }

    reduce = clCreateKernel(program, "reduce_vec", &err);
    checkError(err, "Creating kernel");

    d_a  = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * n, NULL, &err);
    checkError(err, "Creating buffer d_a");

    d_result  = clCreateBuffer(context,  CL_MEM_READ_WRITE,  sizeof(float) * global / local, NULL, &err);
    checkError(err, "Creating buffer d_result");

    err = clEnqueueWriteBuffer(commands, d_a, CL_TRUE, 0, sizeof(float) * n, h_a.data(), 0, NULL, NULL);
    checkError(err, "Copying h_a to device at d_a");

    err  = clSetKernelArg(reduce, 0, sizeof(cl_mem), &d_a);
    err |= clSetKernelArg(reduce, 1, local * sizeof(float), NULL);
    err |= clSetKernelArg(reduce, 2, sizeof(cl_mem), &d_result);
    err |= clSetKernelArg(reduce, 3, sizeof(int), &n);
    checkError(err, "Setting kernel arguments");
    
    prof.tic("vector reduction opencl");
    err = clEnqueueNDRangeKernel(commands, reduce, 1, NULL, &global, &local, 0, NULL, NULL);
    checkError(err, "Enqueueing kernel");

    err = clFinish(commands);
    checkError(err, "Waiting for kernel to finish");

    err = clEnqueueReadBuffer( commands, d_result, CL_TRUE, 0, sizeof(float) * global / local, h_result.data(), 0, NULL, NULL );  
    checkError(err, "Reading results");

    float sum_gpu = std::accumulate(h_result.begin(), h_result.end(), 0.0);

    prof.toc("vector reduction opencl");

  
    prof.tic("reduction cpu");
    float sum_cpu = std::accumulate(h_a.begin(), h_a.end(), 0.0);
    prof.toc("reduction cpu");

    std::cout << "Error: "<< fabs(sum_gpu - sum_cpu) << std::endl;

    // cleanup then shutdown
    clReleaseMemObject(d_a);
    clReleaseMemObject(d_result);
    clReleaseProgram(program);
    clReleaseKernel(reduce);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    prof.report();

    return 0;
}
