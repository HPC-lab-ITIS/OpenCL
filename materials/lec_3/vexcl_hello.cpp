#include <iostream>
#include <stdexcept>
#include <vexcl/vexcl.hpp>

int main() {
    vex::Context ctx( vex::Filter::GPU && vex::Filter::DoublePrecision );

    if (!ctx) throw std::runtime_error("No devices available.");

    std::cout << ctx << std::endl;
}
