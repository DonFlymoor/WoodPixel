#include <iostream>
#include <vector>
//#include <ocl_template_matcher.hpp>
//#include <matching_policies.hpp>
#include <ocl_wrappers.hpp>
#include <string>

using namespace ocl_template_matching::impl;

int main()
{
	try
	{
		auto clplatform{cl::CLState::createInstance(0, 0)};
		std::cout << "Selected platform: \n" << clplatform->get_selected_platform() << "\n";
		std::cout << "Selected device: \n" << clplatform->get_selected_device() << "\n";

		std::string kernel_src = R"(
			kernel void hello_world()
			{
				printf("Hello CL!");
			}
		)";

		auto progra{cl::CLProgram(kernel_src, "", clplatform)};
		cl::CLProgram::ExecParams exparams{
			1,			// Work dimension
			{0, 0, 0},	// Offset
			{1, 0, 0},	// Global work size
			{1, 0, 0}	// Local work size
		};
		progra("hello_world", exparams).wait();
		std::cout << "Weow that worked!";
	}
	catch(const std::exception& ex)
	{
		std::cerr << "[ERROR]" << ex.what() << std::endl;
	}	
	return 0;
}