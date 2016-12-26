#include <iostream>
#include <vector>
#include <algorithm>
#include <vexcl/vexcl.hpp>

int main()
{
    vex::Context ctx( vex::Filter::GPU && vex::Filter::DoublePrecision && vex::Filter::Count(1) );

    const size_t n = 1024 * 1024;
    
    std::vector<double> a(n);
    
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    std::mt19937 engine;
    auto generator = std::bind(distribution, engine);
    std::generate_n(a.begin(), n, generator);
    
    vex::vector<double> A(ctx, a);
    vex::Reductor<double, vex::SUM> sum(ctx);

    std::cout << fabs( sum(A) - std::accumulate(a.begin(), a.end(),0.) )  << std::endl;

    ctx.finish();

    return 0;
}
