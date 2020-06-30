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
#include <memory>
#include <cstdint>

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
			#pragma region context
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

				// instead of a constructor we use a factory function which returns a shared_ptr to the CLState instance
				// (it is a lot safer this way, because the other wrapper classes all depend on a valid CLState.)
				friend std::shared_ptr<CLState> createCLInstance(std::size_t platform_index, std::size_t device_index);

				// dtor
				~CLState();

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

				// ctors
				CLState(std::size_t platform_index, std::size_t device_index);

				// copy / move constructors
				CLState(const CLState&) = delete;
				CLState(CLState&& other) noexcept;

				// copy / move assignment
				CLState& operator=(const CLState&) = delete;
				CLState& operator=(CLState&&) noexcept;

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

			#pragma endregion

			#pragma region program_and_kernels
			// check if T has member funcions to access data pointer and size (for setting kernel params!)
			template <typename T, typename = void>
			struct is_cl_param : public std::false_type	{};

			template <typename T>
			struct is_cl_param <T, ocl_template_matching::meta::void_t<
				decltype(std::size_t{std::declval<const T>().arg_size()}), // has const size() member, returning size_t?,
				typename std::enable_if<std::is_convertible<decltype(std::declval<const T>().arg_data()), const void*>::value>::type // has const arg_data() member returning something convertible to const void* ?
			>> : std::true_type {};

			// traits class for handling kernel arguments
			template <typename T, typename = void>
			struct CLKernelArgTraits;

			// case: complex type which fulfills requirements of is_cl_param<T>
			template <typename T>
			struct CLKernelArgTraits <T, typename std::enable_if<is_cl_param<T>::value>::type>
			{
				static std::size_t arg_size(const T& arg) { return arg.arg_size(); }
				static const void* arg_data(const T& arg) { static_cast<const void*>(return arg.arg_data()) }
			};

			// case: arithmetic type or standard layout type (poc struct, plain array...)
			template <typename T>
			struct CLKernelArgTraits <T, typename std::enable_if<std::is_arithmetic<T>::value || std::is_standard_layout<T>::value>::type>
			{
				static constexpr std::size_t arg_size(const T& arg) { return sizeof(T); }
				static const void* arg_data(const T& arg) { static_cast<const void*>(return &arg) }
			};

			// case: pointer
			template <typename T>
			struct CLKernelArgTraits <T*, void>
			{
				static std::size_t arg_size(const T * const & arg) { return CLKernelArgTraits<T>::arg_size(*arg) }
				static const void* arg_data(const T * const & arg) { return CLKernelArgTraits<T>::arg_data(*arg) }
			};

			// case: nullptr
			template <>
			struct CLKernelArgTraits <std::nullptr_t, void>
			{
				static constexpr std::size_t arg_size(const std::nullptr_t& ptr) { return std::size_t{0}; }
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

				CLProgram(const std::string& source, const std::string& compiler_options, const std::shared_ptr<CLState>& clstate);
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
				template <typename DependencyIterator, typename ... ArgTypes>
				CLEvent operator()(const std::string& name, DependencyIterator start_dep_iterator, DependencyIterator end_dep_iterator, const ExecParams& exec_params, const ArgTypes&... args)
				{
					static_assert(std::is_same<typename std::remove_cv<typename std::remove_reference<typename std::iterator_traits<DependencyIterator>::value_type>::type>::type , CLEvent>::value , "[CLProgram]: Dependency iterators must refer to a collection of CLEvent objects.");
					static_assert(ocl_template_matching::meta::conjunction<is_valid_kernel_arg<ArgTypes>...>::value, "[CLProgram]: Incompatible kernel argument type.");
					try
					{
						// unpack args
						setKernelArgs < std::size_t{0}, ArgTypes... > (name, args...);

						// invoke kernel
						m_event_cache.clear();
						for(DependencyIterator it{start_dep_iterator}; it != end_dep_iterator; ++it)
							m_event_cache.push_back(it->m_event);
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

				template <typename DependencyIterator, typename ... ArgTypes>
				CLEvent operator()(const CLKernelHandle& kernel, DependencyIterator start_dep_iterator, DependencyIterator end_dep_iterator, const ExecParams& exec_params, const ArgTypes&... args)
				{
					static_assert(std::is_same<typename std::remove_cv<typename std::remove_reference<typename std::iterator_traits<DependencyIterator>::value_type>::type>::type, CLEvent>::value, "[CLProgram]: Dependency iterators must refer to a collection of CLEvent objects.");
					static_assert(ocl_template_matching::meta::conjunction<is_valid_kernel_arg<ArgTypes>...>::value, "[CLProgram]: Incompatible kernel argument type.");					
					// unpack args
					setKernelArgs < std::size_t{0}, ArgTypes... > (name, args...);

					// invoke kernel
					m_event_cache.clear();
					for(DependencyIterator it{start_dep_iterator}; it != end_dep_iterator; ++it)
						m_event_cache.push_back(it->m_event);
					return invoke(kernel.m_kernel, m_event_cache, exec_params);					
				}

				// overload for zero arguments (with dependencies)
				template <typename DependencyIterator>
				CLEvent operator()(const std::string& name, DependencyIterator start_dep_iterator, DependencyIterator end_dep_iterator, const ExecParams& exec_params)
				{
					static_assert(std::is_same<typename std::remove_cv<typename std::remove_reference<typename std::iterator_traits<DependencyIterator>::value_type>::type>::type, CLEvent>::value, "[CLProgram]: Dependency iterators must refer to a collection of CLEvent objects.");
					try
					{
						// invoke kernel
						m_event_cache.clear();
						for(DependencyIterator it{start_dep_iterator}; it != end_dep_iterator; ++it)
							m_event_cache.push_back(it->m_event);
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

				template <typename DependencyIterator>
				CLEvent operator()(const CLKernelHandle& kernel, DependencyIterator start_dep_iterator, DependencyIterator end_dep_iterator, const ExecParams& exec_params)
				{				
					static_assert(std::is_same<typename std::remove_cv<typename std::remove_reference<typename std::iterator_traits<DependencyIterator>::value_type>::type>::type, CLEvent>::value, "[CLProgram]: Dependency iterators must refer to a collection of CLEvent objects.");
					// invoke kernel
					m_event_cache.clear();
					for(DependencyIterator it{start_dep_iterator}; it != end_dep_iterator; ++it)
						m_event_cache.push_back(it->m_event);
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
					setKernelArgsImpl(name, index, CLKernelArgTraits<FirstArgType>::arg_size(), CLKernelArgTraits<FirstArgType>::arg_data());
				}

				std::string m_source;
				std::string m_options;
				std::unordered_map<std::string, CLKernel> m_kernels;
				cl_program m_cl_program;
				std::shared_ptr<CLState> m_cl_state;
				std::vector<cl_event> m_event_cache;
			};
			#pragma endregion
		
			#pragma region buffers

			class CLBuffer
			{
			public:
				CLBuffer(std::size_t size, cl_mem_flags flags, const std::shared_ptr<CLState>& clstate, void* hostptr = nullptr);

				~CLBuffer() noexcept;
				CLBuffer(const CLBuffer&) = delete;
				CLBuffer(CLBuffer&& other) noexcept;
				CLBuffer& operator=(const CLBuffer&) = delete;
				CLBuffer& operator=(CLBuffer&& other) noexcept;

				//! Copies data pointed to by data into the OpenCL buffer.
				/*!	
				*	Copies data pointed to by data into the OpenCL buffer. The function returns a CLEvent which can be waited upon. It refers to the unmap command after copying.
				*	Setting invalidate = true invalidates the written buffer region (all data that was not written is now in undefined state!) but most likely increases performance
				*	due to less synchronization overhead in the driver.
				*
				*	\param[in]		data		Points to the data to be written into the buffer.
				*	\param[in]		length		Length of the data to be written in bytes. If 0 (default), the whole buffer will be written and the offset is ignored.
				*	\param[in]		offset		Offset into the buffer where the region to be written begins. Ignored if length is 0.
				*	\param[in]		invalidate	If true, the written region will be invalidated which provides performance benefits in most cases.
				*
				*	\return			Returns a CLEvent object which can be waited upon either by other OpenCL operations or explicitely to block until the data is synchronized with OpenCL.
				*	
				*	\attention		This is a low level function. Please consider using one of the type-safe versions instead. If this function is used directly make sure that access to data*
				*					in the region [data, data + length - 1] does not produce access violations!
				*/
				inline CLEvent write_bytes(const void* data, std::size_t length = 0ull, std::size_t offset = 0ull, bool invalidate = false);

				//! Copies data from the OpenCL buffer into the memory region pointed to by data.
				/*!
				*	Copies data from the OpenCL buffer into the memory region pointed to by data. The function returns a CLEvent which can be waited upon. It refers to the unmap command after copying.
				*
				*	\param[out]		data		Points to the memory region the buffer should be read into.
				*	\param[in]		length		Length of the data to be read in bytes. If 0 (default), the whole buffer will be read and the offset is ignored.
				*	\param[in]		offset		Offset into the buffer where the region to be read begins. Ignored if length is 0.
				*
				*	\return			Returns a CLEvent object which can be waited upon either by other OpenCL operations or explicitely to block until the data is synchronized with OpenCL.
				*
				*	\attention		This is a low level function. Please consider using one of the type-safe versions instead. If this function is used directly make sure that access to data*
				*					in the region [data, data + length - 1] does not produce access violations!
				*/
				inline CLEvent read_bytes(void* data, std::size_t length = 0ull, std::size_t offset = 0ull);

				//! Copies data pointed to by data into the OpenCL buffer after waiting on a list of dependencies (CLEvent's).
				/*!
				*	Copies data pointed to by data into the OpenCL buffer. Before the buffer is mapped for writing, OpenCL waits for the provided CLEvent's.
				*	The function returns a CLEvent which can be waited upon. It refers to the unmap command after copying.
				*	Setting invalidate = true invalidates the written buffer region (all data that was not written is now in undefined state!) but most likely increases performance
				*	due to less synchronization overhead in the driver.
				*
				*	\tparam			DepIterator	Input iterator to iterate over a collection of CLEvent's.
				*
				*	\param[in]		data		Points to the data to be written into the buffer.
				*	\param[in]		dep_begin	Start iterator of a collection of CLEvent's.
				*	\param[in]		dep_end		End iterator of a collection of CLEvent's.
				*	\param[in]		length		Length of the data to be written in bytes. If 0 (default), the whole buffer will be written and the offset is ignored.
				*	\param[in]		offset		Offset into the buffer where the region to be written begins. Ignored if length is 0.
				*	\param[in]		invalidate	If true, the written region will be invalidated which provides performance benefits in most cases.
				*
				*	\return			Returns a CLEvent object which can be waited upon either by other OpenCL operations or explicitely to block until the data is synchronized with OpenCL.
				*
				*	\attention		This is a low level function. Please consider using one of the type-safe versions instead. If this function is used directly make sure that access to data*
				*					in the region [data, data + length - 1] does not produce access violations!
				*/
				template <typename DepIterator>
				inline CLEvent write_bytes(const void* data, DepIterator dep_begin, DepIterator dep_end, std::size_t length = 0ull, std::size_t offset = 0ull, bool invalidate = false);

				//! Copies data from the OpenCL buffer into the memory region pointed to by data after waiting on a list of dependencies (CLEvent's).
				/*!
				*	Copies data from the OpenCL buffer into the memory region pointed to by data. Before the buffer is mapped for reading, OpenCL waits for the provided CLEvent's.
				*	The function returns a CLEvent which can be waited upon. It refers to the unmap command after copying.
				*
				*	\tparam			DepIterator	Input iterator to iterate over a collection of CLEvent's.
				*
				*	\param[out]		data		Points to the memory region the buffer should be read into.
				*	\param[in]		dep_begin	Start iterator of a collection of CLEvent's.
				*	\param[in]		dep_end		End iterator of a collection of CLEvent's.
				*	\param[in]		length		Length of the data to be read in bytes. If 0 (default), the whole buffer will be read and the offset is ignored.
				*	\param[in]		offset		Offset into the buffer where the region to be read begins. Ignored if length is 0.
				*
				*	\return			Returns a CLEvent object which can be waited upon either by other OpenCL operations or explicitely to block until the data is synchronized with OpenCL.
				*
				*	\attention		This is a low level function. Please consider using one of the type-safe versions instead. If this function is used directly make sure that access to data*
				*					in the region [data, data + length - 1] does not produce access violations!
				*/
				template <typename DepIterator>
				inline CLEvent read_bytes(void* data, DepIterator dep_begin, DepIterator dep_end, std::size_t length = 0ull, std::size_t offset = 0ull);

				// high level read / write
				template <typename DataIterator>
				inline CLEvent write(DataIterator data_begin, DataIterator data_end, std::size_t offset = 0ull, bool invalidate = false);

				template <typename DataIterator>
				inline CLEvent read(DataIterator data_begin, std::size_t num_elements, std::size_t offset = 0ull);

				// with dependencies
				template <typename DataIterator, typename DepIterator>
				inline CLEvent write(DataIterator data_begin, DataIterator data_end, DepIterator dep_begin, DepIterator dep_end, std::size_t offset = 0ull, bool invalidate = false);

				template <typename DataIterator, typename DepIterator>
				inline CLEvent read(DataIterator data_begin, std::size_t num_elements, DepIterator dep_begin, DepIterator dep_end, std::size_t offset = 0ull);

				//! reports allocated size
				std::size_t size() const noexcept;

				//! interface used by CLProgram kernel execution
				static constexpr std::size_t arg_size() { return sizeof(cl_mem); }
				const void* arg_data() const { return m_cl_memory; }
				
			private:
				CLEvent buf_write(const void* data, std::size_t length = 0ull, std::size_t offset = 0ull, bool invalidate = false);
				CLEvent buf_read(void* data, std::size_t length = 0ull, std::size_t offset = 0ull) const;

				void* map_buffer(std::size_t length, std::size_t offset, bool write, bool invalidate = false);
				CLEvent unmap_buffer(void* bufptr);

				cl_mem m_cl_memory;
				cl_mem_flags m_flags;
				void* m_hostptr;
				std::size_t m_size;
				std::shared_ptr<CLState> m_cl_state;
				std::vector<cl_event> m_event_cache;
			};

			CLEvent ocl_template_matching::impl::cl::CLBuffer::write_bytes(const void* data, std::size_t length, std::size_t offset, bool invalidate)
			{
				m_event_cache.clear();
				return buf_write(data, length, offset, invalidate);
			}

			CLEvent ocl_template_matching::impl::cl::CLBuffer::read_bytes(void* data, std::size_t length, std::size_t offset)
			{
				m_event_cache.clear();
				return buf_read(data, length, offset);
			}

			template<typename DepIterator>
			inline CLEvent ocl_template_matching::impl::cl::CLBuffer::write_bytes(const void* data, DepIterator dep_begin, DepIterator dep_end, std::size_t length, std::size_t offset, bool invalidate)
			{
				static_assert(std::is_same<typename std::remove_cv<typename std::remove_reference<typename std::iterator_traits<DepIterator>::value_type>::type>::type, CLEvent>::value, "[CLBuffer]: Dependency iterators must refer to a collection of CLEvent objects.");
				m_event_cache.clear();
				for(DepIterator it{dep_begin}; it != dep_end; ++it)
					m_event_cache.push_back(it->m_event);
				return buf_write(data, length, offset, invalidate);
			}

			template<typename DepIterator>
			inline CLEvent ocl_template_matching::impl::cl::CLBuffer::read_bytes(void* data, DepIterator dep_begin, DepIterator dep_end, std::size_t length, std::size_t offset)
			{
				static_assert(std::is_same<typename std::remove_cv<typename std::remove_reference<typename std::iterator_traits<DepIterator>::value_type>::type>::type, CLEvent>::value, "[CLBuffer]: Dependency iterators must refer to a collection of CLEvent objects.");
				m_event_cache.clear();
				for(DepIterator it{dep_begin}; it != dep_end; ++it)
					m_event_cache.push_back(it->m_event);
				return buf_read(data, length, offset);
			}

			template<typename DataIterator>
			inline CLEvent ocl_template_matching::impl::cl::CLBuffer::write(DataIterator data_begin, DataIterator data_end, std::size_t offset, bool invalidate)
			{
				using elem_t = typename std::static_cast<typename std::iterator_traits<DataIterator>::value_type;
				static_assert(std::is_standard_layout<elem_t>::value, "[CLBuffer]: Types read and written from and to OpenCL buffers must have standard layout.");
				std::size_t datasize = static_cast<std::size_t>(data_end - data_begin) * sizeof(elem_t);
				std::size_t bufoffset = offset * sizeof(elem_t);
				if(bufoffset + datasize > m_size)
					throw std::out_of_range("[CLBuffer]: Buffer write failed. Input offset + length out of range.");
				m_event_cache.clear();
				elem_t* bufptr = static_cast<elem_t*>(map_buffer(datasize, bufoffset, true, invalidate));
				std::size_t bufidx = 0;
				for(DataIterator it{data_begin}; it != data_end; ++it)
					bufptr[bufidx++] = *it;
				return unmap_buffer(static_cast<void*>(bufptr));
			}

			template<typename DataIterator>
			inline CLEvent ocl_template_matching::impl::cl::CLBuffer::read(DataIterator data_begin, std::size_t num_elements, std::size_t offset)
			{
				using elem_t = typename std::static_cast<typename std::iterator_traits<DataIterator>::value_type;
				static_assert(std::is_standard_layout<elem_t>::value, "[CLBuffer]: Types read and written from and to OpenCL buffers must have standard layout.");
				std::size_t datasize = num_elements * sizeof(elem_t);
				std::size_t bufoffset = offset * sizeof(elem_t);
				if(bufoffset + datasize > m_size)
					throw std::out_of_range("[CLBuffer]: Buffer read failed. Input offset + length out of range.");
				m_event_cache.clear();
				elem_t* bufptr = static_cast<elem_t*>(map_buffer(datasize, bufoffset, false, false));
				std::size_t bufidx = 0;
				for(DataIterator it{data_begin}; it != data_end; ++it)
					*it = bufptr[bufidx++];
				return unmap_buffer(static_cast<void*>(bufptr));
			}

			template<typename DataIterator, typename DepIterator>
			inline CLEvent ocl_template_matching::impl::cl::CLBuffer::write(DataIterator data_begin, DataIterator data_end, DepIterator dep_begin, DepIterator dep_end, std::size_t offset, bool invalidate)
			{
				static_assert(std::is_same<typename std::remove_cv<typename std::remove_reference<typename std::iterator_traits<DepIterator>::value_type>::type>::type, CLEvent>::value, "[CLBuffer]: Dependency iterators must refer to a collection of CLEvent objects.");
				m_event_cache.clear();
				for(DepIterator it{dep_begin}; it != dep_end; ++it)
					m_event_cache.push_back(it->m_event);
				using elem_t = typename std::static_cast<typename std::iterator_traits<DataIterator>::value_type;
				static_assert(std::is_standard_layout<elem_t>::value, "[CLBuffer]: Types read and written from and to OpenCL buffers must have standard layout.");
				std::size_t datasize = static_cast<std::size_t>(data_end - data_begin) * sizeof(elem_t);
				std::size_t bufoffset = offset * sizeof(elem_t);
				if(bufoffset + datasize > m_size)
					throw std::out_of_range("[CLBuffer]: Buffer write failed. Input offset + length out of range.");
				elem_t* bufptr = static_cast<elem_t*>(map_buffer(datasize, bufoffset, true, invalidate));
				std::size_t bufidx = 0;
				for(DataIterator it{data_begin}; it != data_end; ++it)
					bufptr[bufidx++] = *it;
				return unmap_buffer(static_cast<void*>(bufptr));
			}

			template<typename DataIterator, typename DepIterator>
			inline CLEvent ocl_template_matching::impl::cl::CLBuffer::read(DataIterator data_begin, std::size_t num_elements, DepIterator dep_begin, DepIterator dep_end, std::size_t offset)
			{
				static_assert(std::is_same<typename std::remove_cv<typename std::remove_reference<typename std::iterator_traits<DepIterator>::value_type>::type>::type, CLEvent>::value, "[CLBuffer]: Dependency iterators must refer to a collection of CLEvent objects.");
				m_event_cache.clear();
				for(DepIterator it{dep_begin}; it != dep_end; ++it)
					m_event_cache.push_back(it->m_event);
				using elem_t = typename std::static_cast<typename std::iterator_traits<DataIterator>::value_type;
				static_assert(std::is_standard_layout<elem_t>::value, "[CLBuffer]: Types read and written from and to OpenCL buffers must have standard layout.");
				std::size_t datasize = num_elements * sizeof(elem_t);
				std::size_t bufoffset = offset * sizeof(elem_t);
				if(bufoffset + datasize > m_size)
					throw std::out_of_range("[CLBuffer]: Buffer read failed. Input offset + length out of range.");
				elem_t* bufptr = static_cast<elem_t*>(map_buffer(datasize, bufoffset, false, false));
				std::size_t bufidx = 0;
				for(DataIterator it{data_begin}; it != data_end; ++it)
					*it = bufptr[bufidx++];
				return unmap_buffer(static_cast<void*>(bufptr));
			}

			#pragma endregion

			#pragma region images
			class CLImage
			{
			public:
				enum class ImageType : cl_mem_object_type
				{
					Image1D = CL_MEM_OBJECT_IMAGE1D,
					Image2D = CL_MEM_OBJECT_IMAGE2D,
					Image3D = CL_MEM_OBJECT_IMAGE3D,
					Image1DArray = CL_MEM_OBJECT_IMAGE1D_ARRAY,
					Image2DArray = CL_MEM_OBJECT_IMAGE2D_ARRAY
				};

				enum class ImageChannelOrder : cl_uint
				{
					R = CL_R,
					RG = CL_RG,
					RGBA = CL_RGBA,
					BGRA = CL_BGRA,
					sRGBA = CL_sRGBA
				};

				enum class ImageChannelType : cl_uint
				{
					SNORM_INT8 = CL_SNORM_INT8,
					SNORM_INT16 = CL_SNORM_INT16,
					UNORM_INT8 = CL_UNORM_INT8,
					UNORM_INT_16 = CL_UNORM_INT16,
					INT_8 = CL_SIGNED_INT8,
					INT_16 = CL_SIGNED_INT16,
					INT_32 = CL_SIGNED_INT32,
					UINT_8 = CL_UNSIGNED_INT8,
					UINT_16 = CL_UNSIGNED_INT16,
					UINT_32 = CL_UNSIGNED_INT32,
					HALF = CL_HALF_FLOAT,
					FLOAT = CL_FLOAT
				};

				enum class ImageAccess : cl_mem_flags
				{
					Read = CL_MEM_READ_ONLY,
					Write = CL_MEM_WRITE_ONLY ,
					ReadWrite = CL_MEM_READ_WRITE
				};

				struct ImageDimensions
				{
					ImageDimensions() noexcept: width{0ull}, height{0ull}, depth{0ull}, row_pitch{0ull}, slice_pitch{0ull} {}
					ImageDimensions(std::size_t width = 0ull, std::size_t height = 0ull, std::size_t depth = 0ull,
						std::size_t row_pitch = 0ull, std::size_t slice_pitch = 0ull) noexcept :
						width{width}, height{height}, depth{depth}, row_pitch{row_pitch}, slice_pitch{slice_pitch} {}
					ImageDimensions(const ImageDimensions& other) noexcept = default;
					ImageDimensions(ImageDimensions&& other) noexcept = default;
					ImageDimensions& operator=(const ImageDimensions& other) noexcept = default;
					ImageDimensions& operator=(ImageDimensions&& other) noexcept = default;

					std::size_t width;
					std::size_t height;
					std::size_t depth;
					std::size_t row_pitch;
					std::size_t slice_pitch;
				};

				struct ImageDesc
				{
					ImageType type;
					ImageDimensions dimensions;
					ImageChannelOrder channel_order;
					ImageChannelType channel_type;
					ImageAccess access;
				};

				// for read write stuff
				enum class HostChannel : uint8_t
				{
					R,
					G,
					B,
					A
				};

				enum class HostDataType : uint8_t
				{
					INT8,
					INT16,
					INT32,
					UINT8,
					UINT16,
					UINT32,
					HALF,
					FLOAT
				};

				enum class ChannelDefaultValue : uint8_t
				{
					Zeros,
					Ones
				};

				struct HostChannelOrder
				{
					std::size_t num_channels;
					HostChannel channel_order[4];
				};

				struct HostFormat
				{
					ImageDimensions dimensions;
					HostChannelOrder channel_order;
					HostDataType channel_type;
				};

				CLImage(const std::shared_ptr<CLState>& clstate, const ImageDesc& image_desc);
				~CLImage() noexcept;
				CLImage(const CLImage&) = delete;
				CLImage(CLImage&& other) noexcept;
				CLImage& operator=(const CLImage&) = delete;
				CLImage& operator=(CLImage&& other) noexcept;

				std::size_t width() const;
				std::size_t height() const;
				std::size_t depth() const;
				std::size_t layers() const;

				// read / write functions
				inline CLEvent write(const HostFormat& format, const void* data_ptr, ChannelDefaultValue default_value = ChannelDefaultValue::Zeros);
				inline CLEvent read(const HostFormat& format, void* data_ptr, ChannelDefaultValue default_value = ChannelDefaultValue::Zeros);
				// with dependencies
				template<typename DepIterator>
				CLEvent write(const HostFormat& format, const void* data_ptr, DepIterator dep_begin, DepIterator dep_end, ChannelDefaultValue default_value = ChannelDefaultValue::Zeros);
				template<typename DepIterator>
				CLEvent read(const HostFormat& format, void* data_ptr, DepIterator dep_begin, DepIterator dep_end, ChannelDefaultValue default_value = ChannelDefaultValue::Zeros);

			private:
				CLEvent img_write(const HostFormat& format, const void* data_ptr, ChannelDefaultValue default_value = ChannelDefaultValue::Zeros);
				CLEvent img_read(const HostFormat& format, void* data_ptr, ChannelDefaultValue default_value = ChannelDefaultValue::Zeros);

				cl_mem m_image;			
				ImageDesc m_image_desc;
				std::vector<cl_event> m_event_cache;
				std::shared_ptr<CLState> m_cl_state;
			};

			CLEvent ocl_template_matching::impl::cl::CLImage::write(const HostFormat& format, const void* data_ptr, ChannelDefaultValue default_value)
			{
				m_event_cache.clear();
				return img_write(format, data_ptr, default_value);
			}

			CLEvent ocl_template_matching::impl::cl::CLImage::read(const HostFormat& format, void* data_ptr, ChannelDefaultValue default_value)
			{
				m_event_cache.clear();
				return img_read(format, data_ptr, default_value);
			}

			template<typename DepIterator>
			inline CLEvent ocl_template_matching::impl::cl::CLImage::write(const HostFormat& format, const void* data_ptr, DepIterator dep_begin, DepIterator dep_end, ChannelDefaultValue default_value)
			{
				static_assert(std::is_same<typename std::remove_cv<typename std::remove_reference<typename std::iterator_traits<DepIterator>::value_type>::type>::type, CLEvent>::value, "[CLImage]: Dependency iterators must refer to a collection of CLEvent objects.");
				m_event_cache.clear();
				for(DepIterator it{dep_begin}; it != dep_end; ++it)
					m_event_cache.push_back(it->m_event);
				return img_write(format, data_ptr, default_value);
			}

			template<typename DepIterator>
			inline CLEvent ocl_template_matching::impl::cl::CLImage::read(const HostFormat& format, void* data_ptr, DepIterator dep_begin, DepIterator dep_end, ChannelDefaultValue default_value)
			{
				static_assert(std::is_same<typename std::remove_cv<typename std::remove_reference<typename std::iterator_traits<DepIterator>::value_type>::type>::type, CLEvent>::value, "[CLImage]: Dependency iterators must refer to a collection of CLEvent objects.");
				m_event_cache.clear();
				for(DepIterator it{dep_begin}; it != dep_end; ++it)
					m_event_cache.push_back(it->m_event);
				return img_read(format, data_ptr, default_value);
			}
		#pragma endregion
		}
	}
}

#endif