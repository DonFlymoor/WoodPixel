#include <iostream>
#include <vector>
//#include <ocl_template_matcher.hpp>
//#include <matching_policies.hpp>
#include <simple_cl.hpp>
#include <string>

using namespace simple_cl;

struct blub
{
	int x;
	float u[80];
};

int main()
{
	try
	{
		auto clplatform{cl::Context::createInstance(0, 0)};
		std::cout << "Selected platform: \n" << clplatform->get_selected_platform() << "\n";
		std::cout << "Selected device: \n" << clplatform->get_selected_device() << "\n";
		std::string kernel_src = R"(
			kernel void hello_world(float a, float b, global float* buf)
			{
				size_t gid = get_global_id(0);
				float c = 0.0f;
				for(size_t i = 0; i < 100000; ++i)
					c += a + b;
				buf[gid] = c;
			}
		)";
		constexpr std::size_t num_vals{1000000000};
		std::vector<cl_float> data(num_vals, 0.0f);
		cl::Buffer buffer{
			sizeof(cl_float) * data.size(),
			cl::MemoryFlags{
				cl::DeviceAccess::WriteOnly,
				cl::HostAccess::ReadWrite,
				cl::HostPointerOption::None
			},
			clplatform
		};

		buffer.write(data.begin(), data.end(), 0ull, true).wait();

		auto progra{cl::Program(kernel_src, "", clplatform)};
		cl::Program::ExecParams exparams{
			1,			// Work dimension
			{0, 0, 0},	// Offset
			{num_vals, 0, 0},	// Global work size
			{64, 0, 0}	// Local work size
		};

		float a{5.0f};
		float b{8.0f};

		auto kernel{progra.getKernel("hello_world")};
		std::cout << "Computing...\n";
		progra(kernel, exparams, a, b, buffer).wait();
		std::cout << "Weow that worked!";

		std::cout << "reading result:" << std::endl;
		buffer.read(data.begin(), data.size()).wait();
		std::cout << "done";
	}
	catch(const std::exception& ex)
	{
		std::cerr << "[ERROR]" << ex.what() << std::endl;
	}	
	return 0;
}