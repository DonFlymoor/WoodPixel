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
	// just some template meta programming helpers
	namespace meta
	{
		// support for void_t in case of C++11 and C++14
	#ifdef WOODPIXELS_LANG_FEATURES_VARIADIC_USING_DECLARATIONS
		template <typename...>
		using void_t = void;
	#else
		namespace detail
		{
			template <typename...>
			struct make_void { typedef void type; };
		}
		template <typename... T>
		using void_t = detail::make_void<T...>::type;
	#endif
	}

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

			// check if T has member funcions to access data pointer and size (for setting kernel params!)
			template <typename T, typename = void>
			struct is_cl_param : public std::false_type	{};

			template <typename T>
			struct is_cl_param < T, ocl_template_matching::meta::void_t<
				decltype(std::size_t{std::declval<const T>().size()}), // has const size() member, returning size_t?,
				std::enable_if<std::is_convertible<decltype(std::declval<const T>().arg_data()), const void*>::value>::type // has const arg_data() member returning something convertible to const void* ?
			>> : std::true_type {};

			// traits class for handling kernel arguments
			template <typename T, typename = void>
			struct CLKernelArgTraits;

			// case: complex type which fulfills requirements of is_cl_param<T>
			template <typename T>
			struct CLKernelArgTraits <T, std::enable_if<is_cl_param<T>::value>::type>
			{
				static std::size_t size(const T& arg) { return arg.size(); }
				static const void* arg_data(const T& arg) { static_cast<const void*>(return arg.arg_data()) }
			};

			// case: arithmetic type or standard layout type (poc struct, plain array...)
			template <typename T>
			struct CLKernelArgTraits <T, std::enable_if<std::is_arithmetic<T>::value || std::is_standard_layout<T>::value>::type>
			{
				static constexpr std::size_t size(const T& arg) { return sizeof(T); }
				static const void* arg_data(const T& arg) { static_cast<const void*>(return &arg) }
			};

			// case: pointer
			template <typename T>
			struct CLKernelArgTraits <T*, void>
			{
				static std::size_t size(const T * const & arg) { return CLKernelArgTraits<T>::size(*arg) }
				static const void* arg_data(const T * const & arg) { return CLKernelArgTraits<T>::arg_data(*arg) }
			};

			// case: nullptr
			template <>
			struct CLKernelArgTraits <std::nullptr_t, void>
			{
				static constexpr std::size_t size(const std::nullptr_t& ptr) { return std::size_t{0}; }
				static constexpr const void* arg_data(const std::nullptr_t& ptr) { return nullptr; }
			};

			// TODO: case: smart pointers? maybe later


			// wrapper for opencl kernel objects.
			// should provide:
			//		- convenient compiling and building of kernel programs
			//			- including meaningful compile and linking error reporting
			//		- easy invocation of kernels with specified work group sizes etc.
			//		- convenient passing of kernel parameters

			// TODO: let the invoke and call operators return a future!
			// TODO: design execparams to hold kernel execution dimensions and stuff
			class CLProgram
			{
			public:
				struct ExecParams
				{

				};

				CLProgram(const std::string& source, const std::string& compiler_options, const CLState* clstate);
				~CLProgram();

				// copy / move constructor
				CLProgram(const CLProgram&) = delete;
				CLProgram(CLProgram&&) noexcept;

				// copy / move assignment
				CLProgram& operator=(const CLProgram& other) = delete;
				CLProgram& operator=(CLProgram&& other) noexcept;

				void cleanup() noexcept;

				template <typename ... ArgTypes>
				void operator()(const std::string& name, const ExecParams& exec_params, const ArgTypes&... args)
				{
					// unpack args
					setKernelArgs<0, ArgTypes...>(name, args...);

					// invoke kernel
					invoke(name, exec_params);
				}

				// overload for zero arguments
				void operator()(const std::string& name, const ExecParams& exec_params)
				{
					invoke(name, exec_params);
				}

			private:
				struct CLKernel
				{
					std::size_t id;
					cl_kernel kernel;
				};

				// invoke kernel
				void invoke(const std::string& name, const ExecParams& exec_params)
				{

				}

				// template parameter pack unpacking
				template <std::size_t index, typename FirstArgType, typename ... ArgTypes>
				void setKernelArgs(const std::string& name, const FirstArgType& first_arg, const ArgTypes&... rest)
				{
					// process first_arg
					setKernelArgs<index, FirstArgType>(name, first_arg);
					// unpack next param
					setKernelArgs<index + 1, ArgTypes...>(name, rest...);
				}

				// exit case
				template <std::size_t index, typename FirstArgType>
				void setKernelArgs(const std::string& name, const FirstArgType& first_arg)
				{
					// process argument
					std::size_t arg_size{CLKernelArgTraits<FirstArgType>::size()};
					const void* arg_data_ptr{CLKernelArgTraits<FirstArgType>::arg_data()};

					// set opencl kernel argument
					try
					{
						CL_EX(clSetKernelArg(m_kernelsat(name), static_cast<cl_uint>(index), arg_size, arg_data_ptr));
					}
					catch(const std::out_of_range& ex) // kernel name wasn't found
					{
						throw std::runtime_error("[CLProgram]: Unknown kernel name");
					}
					catch(...)
					{
						throw;
					}
				}

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