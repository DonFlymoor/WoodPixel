#ifndef _OCL_WRAPPERS_HPP_
#define _OCL_WRAPPERS_HPP_

#include <CL/cl.h>
#include <ocl_error.hpp>
#include <string>
#include <iostream>
#include <vector>
#include <sstream>
#include <unordered_map>
#include <future>
#include <atomic>

// compile time definitions
#define OCL_KERNEL_MAX_WORK_DIM 3 // maximum work dim of opencl kernels

namespace ocl_template_matching
{
	// just some template meta programming helpers
	namespace meta
	{
		// support for void_t in case of C++11 and C++14
	#ifdef WOODPIXELS_LANG_FEATURES_VARIADIC_USING_DECLARATIONS
		template <typename...> // only possible with >=C++14
		using void_t = void;
	#else
		namespace detail	// C++11
		{
			template <typename...>
			struct make_void { typedef void type; };
		}
		template <typename... T>
		using void_t = typename detail::make_void<T...>::type;
	#endif

		// this is only available with >=C++17, so we implement that ourselves
		// used to combine multiple boolean traits into one via conjunction
		template <typename...>
		struct conjunction : std::false_type {};
		template <typename Last>
		struct conjunction<Last> : Last {};
		template <typename First, typename ... Rest>
		struct conjunction<First, Rest...> : std::conditional<bool(First::value), conjunction<Rest...>, First> {};
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
			struct is_cl_param <T, ocl_template_matching::meta::void_t<
				decltype(std::size_t{std::declval<const T>().size()}), // has const size() member, returning size_t?,
				typename std::enable_if<std::is_convertible<decltype(std::declval<const T>().arg_data()), const void*>::value>::type // has const arg_data() member returning something convertible to const void* ?
			>> : std::true_type {};

			// traits class for handling kernel arguments
			template <typename T, typename = void>
			struct CLKernelArgTraits;

			// case: complex type which fulfills requirements of is_cl_param<T>
			template <typename T>
			struct CLKernelArgTraits <T, typename std::enable_if<is_cl_param<T>::value>::type>
			{
				static std::size_t size(const T& arg) { return arg.size(); }
				static const void* arg_data(const T& arg) { static_cast<const void*>(return arg.arg_data()) }
			};

			// case: arithmetic type or standard layout type (poc struct, plain array...)
			template <typename T>
			struct CLKernelArgTraits <T, typename std::enable_if<std::is_arithmetic<T>::value || std::is_standard_layout<T>::value>::type>
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

			// general check for allowed argument types. Used to presend meaningful error message wenn invoked with wrong types.
			template <typename T>
			using is_valid_kernel_arg = std::conditional<
				is_cl_param<T>::value ||
				std::is_arithmetic<T>::value ||
				std::is_standard_layout<T>::value ||
				std::is_same<T, std::nullptr_t>::value ||
				std::is_pointer<T>::value
				, std::true_type, std::false_type>;

			class CLProgram
			{
			public:
				struct ExecParams
				{
					std::size_t work_dim;
					std::size_t work_offset[OCL_KERNEL_MAX_WORK_DIM];
					std::size_t global_work_size[OCL_KERNEL_MAX_WORK_DIM];
					std::size_t local_work_size[OCL_KERNEL_MAX_WORK_DIM];
				};

				class CLEvent
				{
					friend class CLProgram;
				public:
					CLEvent(cl_event ev);
					~CLEvent();
					CLEvent(const CLEvent& other);
					CLEvent(CLEvent&& other) noexcept;
					CLEvent& operator=(const CLEvent& other);
					CLEvent& operator=(CLEvent&& other) noexcept;

					void wait() const;
				private:
					cl_event m_event;
				};

				class CLKernelHandle
				{
					friend class CLProgram;
					CLKernelHandle(const CLKernelHandle& other) noexcept = default;
					CLKernelHandle& operator=(const CLKernelHandle& other) noexcept = default;
					~CLKernelHandle() noexcept = default;
				private:
					CLKernelHandle(cl_kernel kernel) noexcept : m_kernel{kernel} {}
					cl_kernel m_kernel;
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

				// no dependencies
				template <typename ... ArgTypes>
				CLEvent operator()(const std::string& name, const ExecParams& exec_params, const ArgTypes&... args)
				{
					static_assert(ocl_template_matching::meta::conjunction<is_valid_kernel_arg<ArgTypes>...>::value, "[CLProgram]: Incompatible kernel argument type.");
					try
					{
						// unpack args
						setKernelArgs<std::size_t{0}, ArgTypes...> (name, args...);

						// invoke kernel
						m_event_cache.clear();
						return invoke(m_kernels.at(name).kernel, m_event_cache, exec_params);
					}				
					catch(const std::out_of_range&) // kernel name wasn't found
					{
						throw std::runtime_error("[CLProgram]: Unknown kernel name");
					}
					catch(...)
					{
						throw;
					}
				}

				template <typename ... ArgTypes>
				CLEvent operator()(const CLKernelHandle& kernel, const ExecParams& exec_params, const ArgTypes&... args)
				{
					static_assert(ocl_template_matching::meta::conjunction<is_valid_kernel_arg<ArgTypes>...>::value, "[CLProgram]: Incompatible kernel argument type.");					
					// unpack args
					setKernelArgs<std::size_t{0}, ArgTypes...>(name, args...);

					// invoke kernel
					m_event_cache.clear();
					return invoke(kernel.m_kernel, m_event_cache, exec_params);					
				}

				// overload for zero arguments (no dependencies)
				CLEvent operator()(const std::string& name, const ExecParams& exec_params)
				{
					try
					{
						// invoke kernel
						m_event_cache.clear();
						return invoke(m_kernels.at(name).kernel, m_event_cache, exec_params);
					}
					catch(const std::out_of_range&) // kernel name wasn't found
					{
						throw std::runtime_error("[CLProgram]: Unknown kernel name");
					}
					catch(...)
					{
						throw;
					}
				}

				CLEvent operator()(const CLKernelHandle& kernel, const ExecParams& exec_params)
				{					
					// invoke kernel
					m_event_cache.clear();
					return invoke(kernel.m_kernel, m_event_cache, exec_params);					
				}

				// call operators with dependencies
				template <typename ... ArgTypes>
				CLEvent operator()(const std::string& name, const std::vector<CLEvent>& dep_events, const ExecParams& exec_params, const ArgTypes&... args)
				{
					static_assert(ocl_template_matching::meta::conjunction<is_valid_kernel_arg<ArgTypes>...>::value, "[CLProgram]: Incompatible kernel argument type.");
					try
					{
						// unpack args
						setKernelArgs < std::size_t{0}, ArgTypes... > (name, args...);

						// invoke kernel
						m_event_cache.clear();
						for(const CLEvent& ev : dep_events)
							m_event_cache.push_back(ev.m_event);
						return invoke(m_kernels.at(name).kernel, m_event_cache, exec_params);
					}
					catch(const std::out_of_range&) // kernel name wasn't found
					{
						throw std::runtime_error("[CLProgram]: Unknown kernel name");
					}
					catch(...)
					{
						throw;
					}
				}

				template <typename ... ArgTypes>
				CLEvent operator()(const CLKernelHandle& kernel, const std::vector<CLEvent>& dep_events, const ExecParams& exec_params, const ArgTypes&... args)
				{
					static_assert(ocl_template_matching::meta::conjunction<is_valid_kernel_arg<ArgTypes>...>::value, "[CLProgram]: Incompatible kernel argument type.");					
					// unpack args
					setKernelArgs < std::size_t{0}, ArgTypes... > (name, args...);

					// invoke kernel
					m_event_cache.clear();
					for(const CLEvent& ev : dep_events)
						m_event_cache.push_back(ev.m_event);
					return invoke(kernel.m_kernel, m_event_cache, exec_params);					
				}

				// overload for zero arguments (with dependencies)
				CLEvent operator()(const std::string& name, const std::vector<CLEvent>& dep_events, const ExecParams& exec_params)
				{
					try
					{
						// invoke kernel
						m_event_cache.clear();
						for(const CLEvent& ev : dep_events)
							m_event_cache.push_back(ev.m_event);
						return invoke(m_kernels.at(name).kernel, m_event_cache, exec_params);
					}
					catch(const std::out_of_range&) // kernel name wasn't found
					{
						throw std::runtime_error("[CLProgram]: Unknown kernel name");
					}
					catch(...)
					{
						throw;
					}
				}

				CLEvent operator()(const CLKernelHandle& kernel, const std::vector<CLEvent>& dep_events, const ExecParams& exec_params)
				{					
					// invoke kernel
					m_event_cache.clear();
					for(const CLEvent& ev : dep_events)
						m_event_cache.push_back(ev.m_event);
					return invoke(kernel.m_kernel, m_event_cache, exec_params);
				}

				// retrieve kernel handle
				CLKernelHandle getKernel(const std::string& name);

			private:
				struct CLKernel
				{
					std::size_t id;
					cl_kernel kernel;
				};

				// invoke kernel
				CLEvent invoke(cl_kernel kernel, const std::vector<cl_event>& dep_events, const ExecParams& exec_params);
				// set kernel params (low level, non type-safe stuff. Implementation hidden in .cpp!)
				void setKernelArgsImpl(const std::string& name, std::size_t index, std::size_t arg_size, const void* arg_data_ptr);

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
					// set opencl kernel argument
					setKernelArgsImpl(name, index, CLKernelArgTraits<FirstArgType>::size(), CLKernelArgTraits<FirstArgType>::arg_data());
				}

				std::string m_source;
				std::string m_options;
				std::unordered_map<std::string, CLKernel> m_kernels;
				cl_program m_cl_program;
				const CLState* m_cl_state;
				std::vector<cl_event> m_event_cache;
			};
		}
	}
}

#endif