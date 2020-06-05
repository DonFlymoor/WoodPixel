#include <CL/cl2.hpp>

#include <iostream>
#include <vector>
#include <cassert>

int main()
{
	// test if opencl works
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	std::cout << "OpenCL Platforms found: " << platforms.size() << std::endl;
	// print platform info
	for(std::size_t p = 0; p < platforms.size(); ++p)
	{
		std::cout << "Platform " << p << ":" << std::endl;
		std::cout << platforms[p].getInfo<CL_PLATFORM_VERSION>();
	}

	return 0;
}