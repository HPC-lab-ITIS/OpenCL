#include <iostream>
#include <vector>
#include <algorithm>
#include <vexcl/vexcl.hpp>

int main()
{
    vex::Context ctx( vex::Filter::GPU && vex::Filter::DoublePrecision && vex::Filter::Count(1) );

    vex::profiler<> prof(ctx);

    prof.tic_cpu("init");
    const size_t n = 256 * 1024 * 1024;

    std::vector<double> a(n);
    std::vector<double> b(n);
    std::vector<double> c_1(n);
    std::vector<double> c_2(n);
    std::vector<double> c_3(n);

    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    std::mt19937 engine;
    auto generator = std::bind(distribution, engine);
    std::generate_n(a.begin(), n, generator);
    std::generate_n(b.begin(), n, generator);
    
    vex::vector<double> A(ctx, a);
    vex::vector<double> B(ctx, b);
    vex::vector<double> C(ctx, n);
    prof.toc("init");

    prof.tic_cpu("cpu");
    for(auto i = 0; i < n; ++i)
        c_1[i] = sqrt(5. * a[i]) + pow( sin(b[i]), 2. );
    prof.toc("cpu");
    
    prof.tic_cpu("vexcl_1");
    C = sqrt(5. * A) + pow(sin(B), 2.);
    vex::copy(C, c_2);
    prof.toc("vexcl_1");

    prof.tic_cpu("vexcl_2");
    VEX_FUNCTION(double, my_func, (double, x)(double, y),
            return sqrt(5. * x) + pow( sin(y), 2. );
            );
    C = my_func(A, B);
    vex::copy(C, c_3);
    prof.toc("vexcl_2");

    for(auto i = 0; i < n; ++i)
        if( fabs(2.*c_1[i] - c_2[i] - c_3[i]) > 1e-5 )
        {
            std::cout << "error" << std::endl;
            break;
        }

    prof.tic_cpu("vexcl_sort");
    vex::sort(C);
    prof.toc("vexcl_sort");

    prof.tic_cpu("cpu_sort");
    std::sort(c_1.begin(), c_1.end());
    prof.toc("cpu_sort");

    ctx.finish();

    std::cout << prof << std::endl;

    return 0;
}
