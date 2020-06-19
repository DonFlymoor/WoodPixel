#ifndef _OCL_WRAPPERS_HPP_
#define _OCL_WRAPPERS_HPP_

#include <CL/cl.h>
#include <ocl_error.hpp>
#include <string>
#include <iostream>
#include <vector>
#include <sstream>
#include <unordered_map>

namespace ocl_template_matching
{
	namespace impl
	{
		namespace util
		{
			std::vector<std::string> string_split(const std::string& s, char delimiter);
			unsigned int get_cl_version_num(const std::string& str);
		}

		namespace cl
		{
			// callbacks
			void create_context_callback(const char* errinfo, const void* private_info, std::size_t cb, void* user_data);

			// wrapper around most important OpenCL State
			class CLState
			{
			public:
				// public types
				struct CLDevice
				{
					cl_device_id device_id;
					cl_uint vendor_id;
					cl_uint max_compute_units;
					cl_uint max_work_item_dimensions;
					std::vector<std::size_t> max_work_item_sizes;
					std::size_t max_work_group_size;
					cl_ulong max_mem_alloc_size;
					std::size_t image2d_max_width;
					std::size_t image2d_max_height;
					std::size_t image3d_max_width;
					std::size_t image3d_max_height;
					std::size_t image3d_max_depth;
					std::size_t image_max_buffer_size;
					std::size_t image_max_array_size;
					cl_uint max_samplers;
					std::size_t max_parameter_size;
					cl_uint mem_base_addr_align;
					cl_uint global_mem_cacheline_size;
					cl_ulong global_mem_cache_size;
					cl_ulong global_mem_size;
					cl_ulong max_constant_buffer_size;
					cl_uint max_constant_args;
					cl_ulong local_mem_size;
					bool little_endian;
					std::string name;
					std::string vendor;
					std::string driver_version;
					std::string device_profile;
					std::string device_version;
					unsigned int device_version_num;
					std::string device_extensions;
					std::size_t printf_buffer_size;
				};
				struct CLPlatform
				{
					cl_platform_id id;
					std::string profile;
					std::string version;
					unsigned int version_num;
					std::string name;
					std::string vendor;
					std::string extensions;
					std::vector<CLDevice> devices;
				};

				// ctors
				CLState(std::size_t platform_index, std::size_t device_index);
				~CLState();

				// copy / move constructors
				CLState(const CLState&) = delete;
				CLState(CLState&& other) noexcept;

				// copy / move assignment
				CLState& operator=(const CLState&) = delete;
				CLState& operator=(CLState&&) noexcept;

				// accessor for context and command queue (return by value because of cl_context and cl_command_queue being pointers)
				cl_context context() const { return m_context; }
				cl_command_queue command_queue() const { return m_command_queue; }

				// for getting device and platform parameters
				const CLPlatform& get_selected_platform() const;
				const CLDevice& get_selected_device() const;

				// print selected platform and device info
				void print_selected_platform_info() const;
				void print_selected_device_info() const;
				// print available platform and device info
				void print_suitable_platform_and_device_info() const;		

			private:
				// --- private types

				struct CLExHolder
				{
					const char* ex_msg;
				};

				// --- private data members

				std::vector<CLPlatform> m_available_platforms;

				// ID's and handles for current OpenCL instance
				std::size_t m_selected_platform_index;
				std::size_t m_selected_device_index;
				cl_context m_context;
				cl_command_queue m_command_queue;

				// If cl error occurs which is supposed to be handled by a callback, we can't throw an exception there.
				// Instead pass a pointer to this member via the "user_data" parameter of the corresponding OpenCL
				// API function.
				CLExHolder m_cl_ex_holder;

				// --- private member functions

				// friends
				// global operators
				friend std::ostream& operator<<(std::ostream&, const CLState::CLPlatform&);
				friend std::ostream& operator<<(std::ostream&, const CLState::CLDevice&);
				// opencl callbacks
				friend void create_context_callback(const char* errinfo, const void* private_info, std::size_t cb, void* user_data);

				// searches for available platforms and devices
				void read_platform_and_device_info();
				// initiates OpenCL, creates context and command queue
				void init_cl_instance(std::size_t platform_id, std::size_t device_id);
				// frees acquired OpenCL resources
				void cleanup();
			};

			// wrapper for opencl kernel objects.
			// should provide:
			//		- convenient compiling and building of kernel programs
			//			- including meaningful compile and linking error reporting
			//		- easy invocation of kernels with specified work group sizes etc.
			//		- convenient passing of kernel parameters

			class CLProgram
			{
			public:
				CLProgram(const std::string& source, const std::string& compiler_options, const CLState* clstate);
				~CLProgram();

				// copy / move constructor
				CLProgram(const CLProgram&) = delete;
				CLProgram(CLProgram&&) noexcept;

				// copy / move assignment
				CLProgram& operator=(const CLProgram& other) = delete;
				CLProgram& operator=(CLProgram&& other) noexcept;

				void cleanup() noexcept;

			private:
				struct CLKernel
				{
					std::size_t id;
					std::string name;
					cl_kernel kernel;
				};

				std::string m_source;
				std::string m_options;
				std::unordered_map<std::string, CLKernel> m_kernels;
				cl_program m_cl_program;
				const CLState* m_cl_state;
			};
		}
	}
}

#endif