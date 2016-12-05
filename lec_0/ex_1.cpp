#define __CL_ENABLE_EXCEPTIONS
#include "CL/cl.hpp"
#include <iostream>
#include <vector>
#include "profiler.h"
#include <algorithm>

const char *ProgramSource =
"__kernel void hello(__global double *input, __global double *output)\n"\
"{\n"\
" size_t id = get_global_id(0);\n"\
" output[id] = input[id] * input[id];\n"\
"}\n";


int checkError(int err, const char *mes)
{
    if(err != CL_SUCCESS)
       std::cout << mes << std::endl;
    return 1;
}

int main(int argc, char** argv)
{
    const size_t n = 1024 * 1024;
    std::vector<double> h_a(n), h_b(n);
    profiler prof;
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    std::mt19937 engine; 
    auto generator = std::bind(distribution, engine);
    std::generate_n(h_a.begin(), n, generator);

    int err;

    size_t global;// global domain size

    cl_device_id device_id;// compute device id
    cl_context context;// compute context
    cl_command_queue commands;// compute command queue
    cl_program program;// compute program
    cl_kernel hello;// compute kernel

    cl_mem d_a;// device memory used for the input  a vector
    cl_mem d_b;// device memory used for the output  b vector

    prof.tic("opencl");
    // Set up platform and GPU device
    cl_platform_id platforms;
    cl_uint num_platforms;
    err = clGetPlatformIDs(1, &platforms, &num_platforms);
    checkError(err, "Finding platforms");
    if (num_platforms == 0)
    {
        printf("Found 0 platforms!\n");
        return 1;
    }

    char buffer[1024];
    clGetPlatformInfo(platforms, CL_PLATFORM_NAME, 1024, buffer, NULL);
    printf("%s", buffer);

    std::cout << std::endl;	

    cl_uint num_of_devices;
    err = clGetDeviceIDs(platforms, CL_DEVICE_TYPE_GPU, 1, &device_id, &num_of_devices);
    if (device_id == NULL)
        checkError(err, "Finding a device");

    clGetDeviceInfo(device_id, CL_DEVICE_NAME, 1024, buffer, NULL);
    printf("%s", buffer);

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
    if (err != CL_SUCCESS)
        return 1;

    // Create the compute kernel from the program
    hello = clCreateKernel(program, "hello", &err);
    checkError(err, "Creating kernel");

    // Create the input (a, b) and output (c) arrays in device memory
    d_a  = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(double) * n, NULL, &err);
    checkError(err, "Creating buffer d_a");

    d_b  = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(double) * n, NULL, &err);
    checkError(err, "Creating buffer d_b");

    // Write a vector into compute device memory
    err = clEnqueueWriteBuffer(commands, d_a, CL_TRUE, 0, sizeof(double) * n, h_a.data(), 0, NULL, NULL);
    checkError(err, "Copying h_a to device at d_a");

    // Set the arguments to our compute kernel
    err  = clSetKernelArg(hello, 0, sizeof(cl_mem), &d_a);
    err |= clSetKernelArg(hello, 1, sizeof(cl_mem), &d_b);
    checkError(err, "Setting kernel arguments");

    // Execute the kernel over the entire range of our 1d input data set
    global = n;
    err = clEnqueueNDRangeKernel(commands, hello, 1, NULL, &global, NULL, 0, NULL, NULL);
    checkError(err, "Enqueueing kernel");

    // Wait for the commands to complete before stopping the timer
    err = clFinish(commands);
    checkError(err, "Waiting for kernel to finish");

    // Read back the results from the compute device
    err = clEnqueueReadBuffer( commands, d_b, CL_TRUE, 0, sizeof(double) * n, h_b.data(), 0, NULL, NULL );  
    if (err != CL_SUCCESS)
        return 1;

    for(auto i = 0; i < n; ++i)
        if(fabs(h_b[i] - h_a[i]*h_a[i]) > 1e-5)
        {
            std::cout << "Error" << std::endl;
            clReleaseMemObject(d_a);
            clReleaseMemObject(d_b);
            clReleaseProgram(program);
            clReleaseKernel(hello);
            clReleaseCommandQueue(commands);
            clReleaseContext(context);
            return 1;
        }

    // cleanup then shutdown
    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseProgram(program);
    clReleaseKernel(hello);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);
    prof.toc("opencl");

    prof.report();

    return 0;
}
