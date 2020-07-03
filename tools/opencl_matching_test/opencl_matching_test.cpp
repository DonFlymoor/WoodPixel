#include <iostream>
#include <vector>
//#include <ocl_template_matcher.hpp>
//#include <matching_policies.hpp>
#include <ocl_wrappers.hpp>
#include <string>

using namespace ocl_template_matching::impl;

struct blub
{
	int x;
	float u[80];
};

int main()
{
	try
	{
		auto clplatform{cl::CLState::createInstance(0, 0)};
		std::cout << "Selected platform: \n" << clplatform->get_selected_platform() << "\n";
		std::cout << "Selected device: \n" << clplatform->get_selected_device() << "\n";
		std::string kernel_src = R"(
			kernel void hello_world(float a, float b)
			{
				printf("Hello CL! %f", a + b);
			}
		)";

		auto progra{cl::CLProgram(kernel_src, "", clplatform)};
		cl::CLProgram::ExecParams exparams{
			1,			// Work dimension
			{0, 0, 0},	// Offset
			{1, 0, 0},	// Global work size
			{1, 0, 0}	// Local work size
		};

		float a{5.0f};
		float b{8.0f};

		auto kernel{progra.getKernel("hello_world")};
		auto event{progra(kernel, exparams, a, b)};
		std::cout << "Weow that worked!";
		event.wait();
	}
	catch(const std::exception& ex)
	{
		std::cerr << "[ERROR]" << ex.what() << std::endl;
	}	
	return 0;
}