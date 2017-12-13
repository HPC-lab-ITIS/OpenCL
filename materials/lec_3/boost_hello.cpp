#include <iostream>
#include <boost/compute/core.hpp>
namespace compute = boost::compute;

int main()
{
    auto platforms = compute::system::platforms();

    for(auto &pl : platforms)
    {
        std::cout << "Platform name: " << pl.name() << std::endl;

        auto devices = pl.devices();
        std::cout << "  Number of devices on the platform: " << devices.size() << std::endl;
        std::cout << "  List of devices: " << std::endl;
        
        int i = 0;
        for(auto &dev : devices)
        {
            std::cout << "      Device #" << i << ": " << dev.name() << std::endl;
            ++i;
        }
    }

    return 0;
}