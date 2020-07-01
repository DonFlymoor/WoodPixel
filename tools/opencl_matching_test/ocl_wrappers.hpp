/** \file ocl_wrappers.h
*	\author Fabian Friederichs
*
*	\brief Provides a minimal set of C++ wrappers for basic OpenCL 1.2 facilities like programs, kernels, buffers and images.
*
*	The classes CLState, CLProgram, CLBuffer and CLImage are declared in this header. CLState abstracts the creation of an OpenCL context, command queue and so on.
*	CLProgram is able to compile OpenCL-C sources and extract all kernel functions which can then be invoked via a type-safe interface.
*	CLBuffer and CLImage allow for simplified creation of buffers and images as well as reading and writing from/to them.
*	CLEvent objects are returned and can be used to synchronize between kernel invokes, write and read operations.
*/

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

/// Maximum work dim of OpenCL kernels
#define OCL_KERNEL_MAX_WORK_DIM 3 // maximum work dim of opencl kernels

/**
*	\namespace ocl_template_matching
*	\brief Contains all the OpenCL template matching functionality.*
*/
namespace ocl_template_matching
{
	/**
	*	\namespace ocl_template_matching::meta
	*	\brief Some template meta programming helpers used e.g. for kernel invocation.
	*/
	namespace meta
	{
	/// enables support for void_t in case of C++11 and C++14
	#ifdef WOODPIXELS_LANG_FEATURES_VARIADIC_USING_DECLARATIONS
		/**
		*	\typedef void_t
		*	\brief Maps an arbitrary set of types to void.
		*	\tparam ...	An arbitrary list of types.
		*/
		template <typename...> // only possible with >=C++14
		using void_t = void;
	#else
		/**
		*	\namespace ocl_template_matching::meta::detail
		*	\brief Encapsulates some implementation detail of the ocl_template_matching::meta namespace
		*/
		namespace detail	// C++11
		{
			/**
			*	\brief Maps an arbitrary set of types to void.
			*	\tparam ...	An arbitrary list of types.
			*/
			template <typename...>
			struct make_void
			{
				typedef void type; ///< void typedef, can be used in template meta programming expressions.
			};
		}
		/**
		*	\typedef void_t
		*	\brief Maps an arbitrary set of types to void.
		*	\tparam ...T	An arbitrary list of types.
		*/
		template <typename... T>
		using void_t = typename detail::make_void<T...>::type;
	#endif

		/**
		*	\brief Conjunction of boolean predicates.
		*	
		*	Exposes a boolean member value. True if all predicates are true, false otherwise.
		*	(std::conjunction is part of the STL since C++17 which would be a pretty restrictive to the users of this library).
		*	\tparam ...	List of predicates.
		*/
		template <typename...>
		struct conjunction : std::false_type {};
		/**
		*	\brief Conjunction of boolean predicates.
		*
		*	Exposes a boolean member value. True if all predicates are true, false otherwise.
		*	(std::conjunction is part of the STL since C++17 which would be a pretty restrictive to the users of this library).
		*	\tparam Last	Last predicate.
		*/
		template <typename Last>
		struct conjunction<Last> : Last {};
		/**
		*	\brief Conjunction of boolean predicates.
		*
		*	Exposes a boolean member value. True if all predicates are true, false otherwise.
		*	(std::conjunction is part of the STL since C++17 which would be a pretty restrictive to the users of this library).
		*	\tparam First	First predicate.
		*	\tparam ...Rest	List of predicates (tail).
		*/
		template <typename First, typename ... Rest>
		struct conjunction<First, Rest...> : std::conditional<bool(First::value), conjunction<Rest...>, First> {};
	}

	/**
	*	\namespace ocl_template_matching::detail
	*	\brief Implementation detail of the ocl_template_matching namespace.
	*/
	namespace impl
	{
		/**
		*	\namespace ocl_template_matching::detail::util
		*	\brief Some utility functions used in this section.
		*/
		namespace util
		{
			/**
			 * \brief	Splits a string around a given delimiter.
			 * \param s String to split.
			 * \param delimiter	Delimiter at which to split the string.
			 * \return Returns a vector of string segments.
			*/
			std::vector<std::string> string_split(const std::string& s, char delimiter);
			/**
			 * \brief Parses an OpenCL version string and returns a numeric expression.
			 * 
			 * E.g. OpenCL 1.2 => 120; OpenCL 2.0 => 200; OpenCL 2.1 => 210...
			 * \param str OpenCL version string to parse.
			 * \return Returns numeric expression. (See examples above)
			*/
			unsigned int get_cl_version_num(const std::string& str);
		}

		/**
		*	\namespace ocl_template_matching::detail::cl
		*	\brief Encapsulates implementation of OpenCL wrappers.
		*/
		namespace cl
		{
			#pragma region context
			/// Callback function used during OpenCL context creation.
			void create_context_callback(const char* errinfo, const void* private_info, std::size_t cb, void* user_data);

			/**
			 *	\brief Creates and manages OpenCL platform, device, context and command queue
			 *
			 *	This class creates the basic OpenCL state needed to run kernels and create buffers and images.
			 *	The constructor is deleted. Please use the factory function createCLInstance(...) instead to retrieve a std::shared_ptr<CLState> to an instance of this class.
			 *	This way the lifetime of the CLState object is ensured to outlive the consuming classes CLBuffer, CLImage and so on.
			*/
			class CLState
			{
			public:
				/**
				 *	\struct	CLDevice
				 *	\brief	Holds information about a device. 
				*/
				struct CLDevice
				{
					cl_device_id device_id;							///< OpenCL device id.
					cl_uint vendor_id;								///< Vendor id.
					cl_uint max_compute_units;						///< Maximum number of compute units on this device.
					cl_uint max_work_item_dimensions;				///< Maximum dimensions of work items. OpenCL compliant GPU's have to provide at least 3.
					std::vector<std::size_t> max_work_item_sizes;	///< Maximum number of work-items that can be specified in each dimension of the work-group.
					std::size_t max_work_group_size;				///< Maximum number of work items per work group executable on a single compute unit.
					cl_ulong max_mem_alloc_size;					///< Maximum number of bytes that can be allocated in a single memory allocation.
					std::size_t image2d_max_width;					///< Maximum width of 2D images.
					std::size_t image2d_max_height;					///< Maximum height of 2D images.
					std::size_t image3d_max_width;					///< Maximum width of 3D images.
					std::size_t image3d_max_height;					///< Maximum height of 3D images.
					std::size_t image3d_max_depth;					///< Maximum depth of 3D images.
					std::size_t image_max_buffer_size;				///< Maximum buffer size for buffer images.
					std::size_t image_max_array_size;				///< Maximum number of array elements for 1D and 2D array images.
					cl_uint max_samplers;							///< Maximum number of samplers that can be used simultaneously in a kernel.
					std::size_t max_parameter_size;					///< Maximum size of parameters (in bytes) assignable to a kernel.
					cl_uint mem_base_addr_align;					///< Alignment requirement (in bits) for sub-buffer offsets. Minimum value is the size of the largest built-in data type supported by the device.
					cl_uint global_mem_cacheline_size;				///< Cache line size of global memory in bytes.
					cl_ulong global_mem_cache_size;					///< Size of global memory cache in bytes.
					cl_ulong global_mem_size;						///< Size of global memory on the device in bytes.
					cl_ulong max_constant_buffer_size;				///< Maximum memory available for constant buffers in bytes.
					cl_uint max_constant_args;						///< Maximum number of __constant arguments for kernels.
					cl_ulong local_mem_size;						///< Size of local memory (per compute unit) on the device in bytes.
					bool little_endian;								///< True if the device is little endian, false otherwise.
					std::string name;								///< Name of the device.
					std::string vendor;								///< Device vendor.
					std::string driver_version;						///< Driver version string.
					std::string device_profile;						///< Device profile. Can be either FULL_PROFILE or EMBEDDED_PROFILE.
					std::string device_version;						///< OpenCL version supported by the device.
					unsigned int device_version_num;				///< Parsed version of the above. 120 => OpenCL 1.2, 200 => OpenCL 2.0...
					std::string device_extensions;					///< Comma-separated list of available extensions supported by this device.
					std::size_t printf_buffer_size;					///< Maximum number of characters printable from a kernel.
				};
				
				/**
				 *	\struct	CLPlatform
				 *	\brief	Holds information about a platform.
				*/
				struct CLPlatform
				{
					cl_platform_id id;					///< OpenCL platform id.
					std::string profile;				///< Supported profile. Can be either FULL_PROFILE or EMBEDDED_PROFILE.
					std::string version;				///< OpenCL version string.
					unsigned int version_num;			///< Parsed version of the above. 120 => OpenCL 1.2, 200 => OpenCL 2.0...
					std::string name;					///< Name of the platform.
					std::string vendor;					///< Platform vendor.
					std::string extensions;				///< Comma-separated list of available extensions supported by this platform.
					std::vector<CLDevice> devices;		///< List of available OpenCL 1.2+ devices on this platform.
				};

				/**
				 * \brief This factory function creates a new instance of CLState and returns a std::shared_ptr<CLState> to this instance.
				 *
				 *	Use this function to create an instance of CLState. The other classes all depend on a valid instance. To ensure the instance outlives
				 *	created CLProgram, CLBuffer and CLImage objects, shared pointers are distributed to these instances.
				 * 
				 *	\param platform_index	Index of the platform to create the context from.
				 *	\param device_index		Index of the device in the selected platform to create the context for.
				 *	\return					A shared pointer to the newly created CLState instance. Use this for instantiating the other wrapper classes.
				*/
				friend std::shared_ptr<CLState> createCLInstance(std::size_t platform_index, std::size_t device_index);

				/// Destructor.
				~CLState();

				/**
				 * \brief	Returns the native OpenCL handle to the context.
				 * \return	Returns the native OpenCL handle to the context.
				*/
				cl_context context() const { return m_context; }
				/**
				 * \brief	Returns the native OpenCL handle to the command queue.
				 * \return  Returns the native OpenCL handle to the command queue.
				*/
				cl_command_queue command_queue() const { return m_command_queue; }

				/**
				 * @brief	Returns the CLPlatform info struct for the selected platform.
				 * @return  Returns the CLDevice info struct for the selected device.
				*/
				const CLPlatform& get_selected_platform() const;
				const CLDevice& get_selected_device() const;

				/**
				 * @brief	Prints detailed information about the selected platform.
				*/
				void print_selected_platform_info() const;
				/**
				 * @brief	Prints detailed infomation about the selected device.
				*/
				void print_selected_device_info() const;
				/**
				 * @brief	Prints detailed information about all suitable (OpenCL 1.2+) platforms and devices available on the system.
				*/
				void print_suitable_platform_and_device_info() const;		

			private:
				/**
				 * \brief Used to retrieve exception information from native OpenCL callbacks.
				*/
				struct CLExHolder
				{
					const char* ex_msg;
				};

				/**
				 * \brief	Constructs context and command queue for the given platform and device index.
				 * \param platform_index	Selected platform index.
				 * \param device_index		Selected device index.
				*/
				CLState(std::size_t platform_index, std::size_t device_index);

				/// No copies are allowed.
				CLState(const CLState&) = delete;
				/// Move the entire state to a new instance.
				CLState(CLState&& other) noexcept;

				/// No copies are allowed.
				CLState& operator=(const CLState&) = delete;
				/// Moves the entire state from one instance to another.
				CLState& operator=(CLState&&) noexcept;

				/// List of available platforms which contain suitable (OpenCL 1.2+) devices.
				std::vector<CLPlatform> m_available_platforms;

				// ID's and handles for current OpenCL instance
				std::size_t m_selected_platform_index;	///< Selected platform index for this instance.
				std::size_t m_selected_device_index;	///< Selected device index for this instance.
				cl_context m_context;					///< OpenCL context handle.
				cl_command_queue m_command_queue;		///< OpenCL command queue handle.

				/**
				* If cl error occurs which is supposed to be handled by a callback, we can't throw an exception there.
				* Instead pass a pointer to this member via the "user_data" parameter of the corresponding OpenCL
				* API function.
				*/
				CLExHolder m_cl_ex_holder;

				// --- private member functions

				// friends
				// global operators
				/// Prints detailed information about the platform.
				friend std::ostream& operator<<(std::ostream&, const CLState::CLPlatform&);
				/// Prints detailed information about the device.
				friend std::ostream& operator<<(std::ostream&, const CLState::CLDevice&);
				// opencl callbacks
				/// Callback used while creating the context.
				friend void create_context_callback(const char* errinfo, const void* private_info, std::size_t cb, void* user_data);

				/**
				 * \brief Searches for available platforms and devices and stores suitable ones (OpenCL 1.2+) in the platforms list member. 
				*/
				void read_platform_and_device_info();
				/**
				 * \brief Initializes OpenCL context and command queue.
				 * \param platform_id Selected platform index.
				 * \param device_id Selected device index.
				*/
				void init_cl_instance(std::size_t platform_id, std::size_t device_id);
				/**
				 * \brief Frees acquired OpenCL resources.
				*/
				void cleanup();
			};

			#pragma endregion

			#pragma region program_and_kernels
			// check if a complex type T has member funcions to access data pointer and size (for setting kernel params!)
			/**
			*	\brief Checks if a complex type T is usable as parameter for CLProgram. Negative case.
			*/
			template <typename T, typename = void>
			struct is_cl_param : public std::false_type	{};

			/**
			*	\brief Checks if a complex type T is usable as parameter for CLProgram. Positive case.
			*
			*	Requirements:
			*	1.	The type has to expose a member std::size_t arg_size() (possibly const) which returns the size in bytes of the param.
			*	2.	The type has to expose a member const void* arg_data() (possibly const) which returns a pointer to arg_size() bytes of data to pass to the kernel as argument.
			*/
			template <typename T>
			struct is_cl_param <T, ocl_template_matching::meta::void_t<
				decltype(std::size_t{std::declval<const T>().arg_size()}), // has const size() member, returning size_t?,
				typename std::enable_if<std::is_convertible<decltype(std::declval<const T>().arg_data()), const void*>::value>::type // has const arg_data() member returning something convertible to const void* ?
			>> : std::true_type {};

			// traits class for handling kernel arguments
			/**
			*	\brief Traits class for convenient processing of kernel arguments. Base template.
			*/
			template <typename T, typename = void>
			struct CLKernelArgTraits;

			// case: complex type which fulfills requirements of is_cl_param<T>
			/**
			*	\brief Traits class for convenient processing of kernel arguments. T fulfills requirements of is_cl_param<T>.
			*	\tparam T type to check for suitability as a kernel argument.
			*/
			template <typename T>
			struct CLKernelArgTraits <T, typename std::enable_if<is_cl_param<T>::value>::type>
			{
				static std::size_t arg_size(const T& arg) { return arg.arg_size(); }
				static const void* arg_data(const T& arg) { static_cast<const void*>(return arg.arg_data()) }
			};

			// case: arithmetic type or standard layout type (poc struct, plain array...)
			/**
			*	\brief Traits class for convenient processing of kernel arguments. T is arithmetic type or has standard layout (POC object!).
			*	\tparam T type to check for suitability as a kernel argument.
			*/
			template <typename T>
			struct CLKernelArgTraits <T, typename std::enable_if<std::is_arithmetic<T>::value || std::is_standard_layout<T>::value>::type>
			{
				static constexpr std::size_t arg_size(const T& arg) { return sizeof(T); }
				static const void* arg_data(const T& arg) { static_cast<const void*>(return &arg) }
			};

			// case: pointer
			/**
			*	\brief Traits class for convenient processing of kernel arguments. Pointer to some argument type.
			*	\tparam T type to check for suitability as a kernel argument.
			*/
			template <typename T>
			struct CLKernelArgTraits <T*, void>
			{
				static std::size_t arg_size(const T * const & arg) { return CLKernelArgTraits<T>::arg_size(*arg) }
				static const void* arg_data(const T * const & arg) { return CLKernelArgTraits<T>::arg_data(*arg) }
			};

			// case: nullptr
			template <>
			/**
			*	\brief Traits class for convenient processing of kernel arguments. Case: nullptr.
			*/
			struct CLKernelArgTraits <std::nullptr_t, void>
			{
				static constexpr std::size_t arg_size(const std::nullptr_t& ptr) { return std::size_t{0}; }
				static constexpr const void* arg_data(const std::nullptr_t& ptr) { return nullptr; }
			};

			// general check for allowed argument types. Used to present meaningful error message wenn invoked with wrong types.
			/**
			*	\brief General check for allowed argument types. Used to present meaningful error message wenn invoked with wrong types.
			*/
			template <typename T>
			using is_valid_kernel_arg = std::conditional<
				is_cl_param<T>::value ||
				std::is_arithmetic<T>::value ||
				std::is_standard_layout<T>::value ||
				std::is_same<T, std::nullptr_t>::value ||
				std::is_pointer<T>::value
				, std::true_type, std::false_type>;

			/**
			 * \brief Handle to some OpenCL event. Can be used to synchronize OpenCL operations.
			*/
			class CLEvent
			{
				friend class CLProgram;
			public:
				/**
				 * \brief Constructs a new handle.
				 * \param ev OpenCL event to encapsulate.
				*/
				CLEvent(cl_event ev);
				/// Destructor. Internally decreases reference count to the cl_event object.
				~CLEvent();
				/// Copy constructor. Internally increases reference count to the cl_event object.
				CLEvent(const CLEvent& other);
				/// Move constructor.
				CLEvent(CLEvent&& other) noexcept;
				/// Copy assignment. Internally increases reference count to the cl_event object.
				CLEvent& operator=(const CLEvent& other);
				/// Moce assignment.
				CLEvent& operator=(CLEvent&& other) noexcept;

				/**
				 * \brief Blocks until the corresponding OpenCL command submitted to the command queue finished execution.
				*/
				void wait() const;
			private:
				cl_event m_event; ///< Handled cl_event object.
			};

			/**
			 * \brief Compiles OpenCL-C source code and extracts kernel functions from this source. Found kernels can then be conveniently invoked using the call operator.
			*/
			class CLProgram
			{
			public:
				/// Defines the global and local dimensions of the kernel invocation in terms of dimensions (up to 3) and work items.
				struct ExecParams
				{
					std::size_t work_dim; ///< Dimension of the work groups and the global work volume. Can be 1, 2 or 3.
					std::size_t work_offset[OCL_KERNEL_MAX_WORK_DIM]; ///< Global offset from the origin.
					std::size_t global_work_size[OCL_KERNEL_MAX_WORK_DIM]; ///< Global work volume dimensions.
					std::size_t local_work_size[OCL_KERNEL_MAX_WORK_DIM]; ///< Local work group dimensions.
				};				

				/**
				 * \brief Handle to an OpenCL kernel in this program. Useful to circumvent kernel name lookup to improve performance of invokes.
				 * \attention This is a non owning handle which becomes invalid if the creating CLProgram instance dies.
				*/
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

				/**
				 * \brief	Compiles OpenCL-C source code, creates a cl_program object and extracts all the available kernel functions.
				 * \param source String containing the entire source code.
				 * \param compiler_options String containing compiler options.
				 * \param clstate A valid CLState intance used to interface with OpenCL.
				*/
				CLProgram(const std::string& source, const std::string& compiler_options, const std::shared_ptr<CLState>& clstate);
				/// Destructor. Frees created cl_program and cl_kernel objects.
				~CLProgram();

				// copy / move constructor
				/// Copy construction is not allowed.
				CLProgram(const CLProgram&) = delete;
				/// Moves the entire state into a new instance.
				CLProgram(CLProgram&&) noexcept;

				// copy / move assignment
				/// Copy assignment is not allowed.
				CLProgram& operator=(const CLProgram& other) = delete;
				/// Moves the entire state into another instance.
				CLProgram& operator=(CLProgram&& other) noexcept;

				// no dependencies
				/**
				*	\brief Invokes the kernel 'name' with execution parameters 'exec_params' and passes an arbitrary list of arguments.
				*
				*	All argument types have to satisfy is_valid_kernel_arg<T>.
				*	After submitting the kernel invocation onto the command queue, a CLEvent is returned which can be waited on to achieve blocking behaviour
				*	or passed to other OpenCL wrapper operations to accomplish synchronization with the following operation.
				*
				*	\tparam ...Argtypes	List of argument types.
				*	\param name Name of the kernel function to invoke.
				*	\param exec_params	Defines execution dimensions of global work volume and local work groups for this invocation.
				*	\param args	List of arguments to pass to the kernel.
				*	\return CLEvent object. Calling wait() on this object blocks until the kernel has finished execution.
				*/
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

				/**
				*	\brief Invokes the kernel 'kernel' with execution parameters 'exec_params' and passes an arbitrary list of arguments.
				*
				*	This overload bypasses the kernel name lookup which can be beneficial in terms of invocation overhead.
				*	All argument types have to satisfy is_valid_kernel_arg<T>.
				*	After submitting the kernel invocation onto the command queue, a CLEvent is returned which can be waited on to achieve blocking behaviour
				*	or passed to other OpenCL wrapper operations to accomplish synchronization with the following operation.
				*
				*	\tparam ...Argtypes	List of argument types.
				*	\param kernel Handle of the kernel function to invoke.
				*	\param exec_params	Defines execution dimensions of global work volume and local work groups for this invocation.
				*	\param args	List of arguments to pass to the kernel.
				*	\return CLEvent object. Calling wait() on this object blocks until the kernel has finished execution.
				*/
				template <typename ... ArgTypes>
				CLEvent operator()(const CLKernelHandle& kernel, const ExecParams& exec_params, const ArgTypes&... args)
				{
					static_assert(ocl_template_matching::meta::conjunction<is_valid_kernel_arg<ArgTypes>...>::value, "[CLProgram]: Incompatible kernel argument type.");					
					// unpack args
					setKernelArgs<std::size_t{0}, ArgTypes...>(kernel.m_kernel, args...);

					// invoke kernel
					m_event_cache.clear();
					return invoke(kernel.m_kernel, m_event_cache, exec_params);					
				}

				// overload for zero arguments (no dependencies)
				/**
				*	\brief Invokes the kernel 'name' with execution parameters 'exec_params'.
				*
				*	No arguments are passed with this overload.
				*	After submitting the kernel invocation onto the command queue, a CLEvent is returned which can be waited on to achieve blocking behaviour
				*	or passed to other OpenCL wrapper operations to accomplish synchronization with the following operation.
				*
				*	\param name Name of the kernel function to invoke.
				*	\param exec_params	Defines execution dimensions of global work volume and local work groups for this invocation.
				*	\return CLEvent object. Calling wait() on this object blocks until the kernel has finished execution.
				*/
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

				/**
				*	\brief Invokes the kernel 'kernel' with execution parameters 'exec_params'.
				*
				*	This overload bypasses the kernel name lookup which can be beneficial in terms of invocation overhead.
				*	No arguments are passed with this overload.
				*	After submitting the kernel invocation onto the command queue, a CLEvent is returned which can be waited on to achieve blocking behaviour
				*	or passed to other OpenCL wrapper operations to accomplish synchronization with the following operation.
				*
				*	\param kernel Handle of the kernel function to invoke.
				*	\param exec_params	Defines execution dimensions of global work volume and local work groups for this invocation.
				*	\return CLEvent object. Calling wait() on this object blocks until the kernel has finished execution.
				*/
				CLEvent operator()(const CLKernelHandle& kernel, const ExecParams& exec_params)
				{					
					// invoke kernel
					m_event_cache.clear();
					return invoke(kernel.m_kernel, m_event_cache, exec_params);					
				}

				// call operators with dependencies
				/**
				*	\brief Invokes the kernel 'name' with execution parameters 'exec_params' and passes an arbitrary list of arguments after waiting for a collection of CLEvents.
				*
				*	All argument types have to satisfy is_valid_kernel_arg<T>.
				*	The kernel waits for finalization of the passed events before it proceeds with its own execution.
				*	After submitting the kernel invocation onto the command queue, a CLEvent is returned which can be waited on to achieve blocking behaviour
				*	or passed to other OpenCL wrapper operations to accomplish synchronization with the following operation.
				*
				*	\tparam ...Argtypes	List of argument types.
				*	\tparam DependencyIterator Iterator which refers to a collection of CLEvent's.
				*	\param name Name of the kernel function to invoke.
				*	\param start_dep_iterator Start iterator of the event collection.
				*	\param end_dep_iterator End iterator of the event collection.
				*	\param exec_params	Defines execution dimensions of global work volume and local work groups for this invocation.
				*	\param args	List of arguments to pass to the kernel.
				*	\return CLEvent object. Calling wait() on this object blocks until the kernel has finished execution.
				*/
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

				/**
				*	\brief Invokes the kernel 'kernel' with execution parameters 'exec_params' and passes an arbitrary list of arguments after waiting for a collection of CLEvents.
				*
				*	This overload bypasses the kernel name lookup which can be beneficial in terms of invocation overhead.
				*	All argument types have to satisfy is_valid_kernel_arg<T>.
				*	The kernel waits for finalization of the passed events before it proceeds with its own execution.
				*	After submitting the kernel invocation onto the command queue, a CLEvent is returned which can be waited on to achieve blocking behaviour
				*	or passed to other OpenCL wrapper operations to accomplish synchronization with the following operation.
				*
				*	\tparam ...Argtypes	List of argument types.
				*	\tparam DependencyIterator Iterator which refers to a collection of CLEvent's.
				*	\param kernel Handle of the kernel function to invoke.
				*	\param start_dep_iterator Start iterator of the event collection.
				*	\param end_dep_iterator End iterator of the event collection.
				*	\param exec_params	Defines execution dimensions of global work volume and local work groups for this invocation.
				*	\param args	List of arguments to pass to the kernel.
				*	\return CLEvent object. Calling wait() on this object blocks until the kernel has finished execution.
				*/
				template <typename DependencyIterator, typename ... ArgTypes>
				CLEvent operator()(const CLKernelHandle& kernel, DependencyIterator start_dep_iterator, DependencyIterator end_dep_iterator, const ExecParams& exec_params, const ArgTypes&... args)
				{
					static_assert(std::is_same<typename std::remove_cv<typename std::remove_reference<typename std::iterator_traits<DependencyIterator>::value_type>::type>::type, CLEvent>::value, "[CLProgram]: Dependency iterators must refer to a collection of CLEvent objects.");
					static_assert(ocl_template_matching::meta::conjunction<is_valid_kernel_arg<ArgTypes>...>::value, "[CLProgram]: Incompatible kernel argument type.");					
					// unpack args
					setKernelArgs < std::size_t{0}, ArgTypes... > (kernel.m_kernel, args...);

					// invoke kernel
					m_event_cache.clear();
					for(DependencyIterator it{start_dep_iterator}; it != end_dep_iterator; ++it)
						m_event_cache.push_back(it->m_event);
					return invoke(kernel.m_kernel, m_event_cache, exec_params);					
				}

				// overload for zero arguments (with dependencies)
				/**
				*	\brief Invokes the kernel 'name' with execution parameters 'exec_params' after waiting for a collection of CLEvents.
				*
				*	The kernel waits for finalization of the passed events before it proceeds with its own execution.
				*	After submitting the kernel invocation onto the command queue, a CLEvent is returned which can be waited on to achieve blocking behaviour
				*	or passed to other OpenCL wrapper operations to accomplish synchronization with the following operation.
				*
				*	\tparam DependencyIterator Iterator which refers to a collection of CLEvent's.
				*	\param name Name of the kernel function to invoke.
				*	\param start_dep_iterator Start iterator of the event collection.
				*	\param end_dep_iterator End iterator of the event collection.
				*	\param exec_params	Defines execution dimensions of global work volume and local work groups for this invocation.
				*	\return CLEvent object. Calling wait() on this object blocks until the kernel has finished execution.
				*/
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

				/**
				*	\brief Invokes the kernel 'kernel' with execution parameters 'exec_params' after waiting for a collection of CLEvents.
				*
				*	This overload bypasses the kernel name lookup which can be beneficial in terms of invocation overhead.
				*	The kernel waits for finalization of the passed events before it proceeds with its own execution.
				*	After submitting the kernel invocation onto the command queue, a CLEvent is returned which can be waited on to achieve blocking behaviour
				*	or passed to other OpenCL wrapper operations to accomplish synchronization with the following operation.
				*
				*	\tparam DependencyIterator Iterator which refers to a collection of CLEvent's.
				*	\param kernel Handle of the kernel function to invoke.
				*	\param start_dep_iterator Start iterator of the event collection.
				*	\param end_dep_iterator End iterator of the event collection.
				*	\param exec_params	Defines execution dimensions of global work volume and local work groups for this invocation.
				*	\return CLEvent object. Calling wait() on this object blocks until the kernel has finished execution.
				*/
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
				/**
				*	\brief Returns a kernel handle to the kernel with name name.
				*	\param name	Name of the kernel to create a handle of.
				*/
				CLKernelHandle getKernel(const std::string& name);

			private:
				/// Cleans up internal state.
				void cleanup() noexcept;

				/**
				 * \brief Holds running id and OpenCL kernel object handle.
				*/
				struct CLKernel
				{
					std::size_t id;		///< Running id
					cl_kernel kernel;	///< OpenCL kernel object handle
				};

				// invoke kernel
				/**
				*	\brief invokes the kernel.
				*	\param kernel	OpenCL kernel object handle
				*	\param dep_events	Vector of events to wait for. (std::vector because we need them in contiguous memory for the API call)
				*	\param exec_params	Execution dimensions.
				*/
				CLEvent invoke(cl_kernel kernel, const std::vector<cl_event>& dep_events, const ExecParams& exec_params);
				// set kernel params (low level, non type-safe stuff. Implementation hidden in .cpp!)
				/**
				*	\brief Sets kernel arguments in a low-level fashion.
				*	\attention This function is not type safe. Use the high level functions above instead!
				*	\param name Name of the kernel.
				*/
				void setKernelArgsImpl(const std::string& name, std::size_t index, std::size_t arg_size, const void* arg_data_ptr);
				/**
				*	\brief Sets kernel arguments in a low-level fashion.
				*	\attention This function is not type safe. Use the high level functions above instead!
				*	\param kernel	OpenCL kernel object handle.
				*/
				void setKernelArgsImpl(cl_kernel kernel, std::size_t index, std::size_t arg_size, const void* arg_data_ptr);

				// template parameter pack unpacking
				/**
				*	\brief	Unpacks and sets an arbitrary kernel argument list.
				*	\tparam index	Index of the first argument of the list.
				*	\tparam FirstArgType Type of the first argument.
				*	\tparam ...ArgTypes	List of kernel argument types (tail).
				*	\param name	Name of the kernel.
				*	\param first_arg First argument.
				*	\param rest	Rest of arguments (tail).
				*/
				template <std::size_t index, typename FirstArgType, typename ... ArgTypes>
				void setKernelArgs(const std::string& name, const FirstArgType& first_arg, const ArgTypes&... rest)
				{
					// process first_arg
					setKernelArgs<index, FirstArgType>(name, first_arg);
					// unpack next param
					setKernelArgs<index + 1, ArgTypes...>(name, rest...);
				}

				// exit case
				/**
				*	\brief	Unpacks and sets a single kernel argument.
				*	\tparam index Index of the kernel argument.
				*	\tparam FirstArgType Type of the argument.
				*	\param name	Name of the kernel.
				*	\param first_arg Argument.
				*/
				template <std::size_t index, typename FirstArgType>
				void setKernelArgs(const std::string& name, const FirstArgType& first_arg)
				{
					// set opencl kernel argument
					setKernelArgsImpl(name, index, CLKernelArgTraits<FirstArgType>::arg_size(), CLKernelArgTraits<FirstArgType>::arg_data());
				}

				// template parameter pack unpacking
				/**
				*	\brief	Unpacks and sets an arbitrary kernel argument list.
				*	\tparam index Index of the first argument of the list.
				*	\tparam FirstArgType Type of the first argument.
				*	\tparam ...ArgTypes	List of kernel argument types (tail).
				*	\param kernel OpenCL kernel object handle.
				*	\param first_arg First argument.
				*	\param rest	Rest of arguments (tail).
				*/
				template <std::size_t index, typename FirstArgType, typename ... ArgTypes>
				void setKernelArgs(cl_kernel kernel, const FirstArgType& first_arg, const ArgTypes&... rest)
				{
					// process first_arg
					setKernelArgs<index, FirstArgType>(kernel, first_arg);
					// unpack next param
					setKernelArgs<index + 1, ArgTypes...>(kernel, rest...);
				}

				// exit case
				/**
				*	\brief	Unpacks and sets a single kernel argument.
				*	\tparam index Index of the kernel argument.
				*	\tparam FirstArgType Type of the argument.
				*	\param kernel OpenCL kernel object handle.
				*	\param first_arg Argument.
				*/
				template <std::size_t index, typename FirstArgType>
				void setKernelArgs(cl_kernel kernel, const FirstArgType& first_arg)
				{
					// set opencl kernel argument
					setKernelArgsImpl(kernel, index, CLKernelArgTraits<FirstArgType>::arg_size(), CLKernelArgTraits<FirstArgType>::arg_data());
				}

				std::string m_source;	///< OpenCL program source code.
				std::string m_options;	///< OpenCL-C compiler options string.
				std::unordered_map<std::string, CLKernel> m_kernels;	///< Map of kernels found in the program, keyed by kernel name.
				cl_program m_cl_program;	///< OpenCL program object handle
				std::shared_ptr<CLState> m_cl_state;	///< Shared pointer to some valid CLState instance.
				std::vector<cl_event> m_event_cache;	///< Used for caching lists of events in contiguous memory.
			};
			#pragma endregion
		
			#pragma region buffers

			/**
			 * \brief Encapsulates creation and read / write operations on OpenCL buffer objects.
			*/
			class CLBuffer
			{
			public:
				CLBuffer(std::size_t size, cl_mem_flags flags, const std::shared_ptr<CLState>& clstate, void* hostptr = nullptr);

				~CLBuffer() noexcept;
				CLBuffer(const CLBuffer&) = delete;
				CLBuffer(CLBuffer&& other) noexcept;
				CLBuffer& operator=(const CLBuffer&) = delete;
				CLBuffer& operator=(CLBuffer&& other) noexcept;

				/** 
				*	\brief Copies data pointed to by data into the OpenCL buffer.
				*	
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

				/**
				*	Copies data from the OpenCL buffer into the memory region pointed to by data.
				*
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

				/** 
				*	Copies data pointed to by data into the OpenCL buffer after waiting on a list of dependencies (CLEvent's).
				*
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

				/**
				*	Copies data from the OpenCL buffer into the memory region pointed to by data after waiting on a list of dependencies (CLEvent's).
				*
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
				/**
				*	\brief Writes some collection of 
				*/
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
			// TODO: Implement reading and writing with non-matching host vs. image channel order and data type
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

				// some bit level hacking to encode number of channels, channel data type size and RGBA-component order.
				// [ 32 bit CL constant | 8 bit channel count | 4 bit R index | 4 bit G index | 4 bit B index | 4 bit A index | 8 bit unused ]
				enum class ImageChannelOrder : uint64_t
				{
					R		= (uint64_t{CL_R} << 32)		| (uint64_t{1} << 24) | (uint64_t{0} << 20) | (uint64_t{0} << 16) | (uint64_t{0} << 12) | (uint64_t{0} << 8),
					RG		= (uint64_t{CL_RG} << 32)		| (uint64_t{2} << 24) | (uint64_t{0} << 20) | (uint64_t{1} << 16) | (uint64_t{0} << 12) | (uint64_t{0} << 8),
					RGBA	= (uint64_t{CL_RGBA} << 32)		| (uint64_t{4} << 24) | (uint64_t{0} << 20) | (uint64_t{1} << 16) | (uint64_t{2} << 12) | (uint64_t{3} << 8),
					BGRA	= (uint64_t{CL_BGRA} << 32)		| (uint64_t{4} << 24) | (uint64_t{2} << 20) | (uint64_t{1} << 16) | (uint64_t{0} << 12) | (uint64_t{3} << 8),
					sRGBA	= (uint64_t{CL_sRGBA} << 32)	| (uint64_t{4} << 24) | (uint64_t{0} << 20) | (uint64_t{1} << 16) | (uint64_t{2} << 12) | (uint64_t{3} << 8)
				};

				// [ 32 bit CL constant | 32 bit size of data type in bytes ]
				enum class ImageChannelType : uint64_t
				{
					SNORM_INT8		= (uint64_t{CL_SNORM_INT8} << 32)		| uint64_t{1},
					SNORM_INT16		= (uint64_t{CL_SNORM_INT16} << 32)		| uint64_t{2},
					UNORM_INT8		= (uint64_t{CL_UNORM_INT8} << 32)		| uint64_t{1},
					UNORM_INT16		= (uint64_t{CL_UNORM_INT16} << 32)		| uint64_t{2},
					INT8			= (uint64_t{CL_SIGNED_INT8} << 32)		| uint64_t{1},
					INT16			= (uint64_t{CL_SIGNED_INT16} << 32)		| uint64_t{2},
					INT32			= (uint64_t{CL_SIGNED_INT32} << 32)		| uint64_t{4},
					UINT8			= (uint64_t{CL_UNSIGNED_INT8} << 32)	| uint64_t{1},
					UINT16			= (uint64_t{CL_UNSIGNED_INT16} << 32)	| uint64_t{2},
					UINT32			= (uint64_t{CL_UNSIGNED_INT32} << 32)	| uint64_t{4},
					HALF			= (uint64_t{CL_HALF_FLOAT} << 32)		| uint64_t{2},
					FLOAT			= (uint64_t{CL_FLOAT} << 32)			| uint64_t{4}
				};

				enum class ImageAccess : cl_mem_flags
				{
					Read = CL_MEM_READ_ONLY,
					Write = CL_MEM_WRITE_ONLY ,
					ReadWrite = CL_MEM_READ_WRITE
				};

				struct ImageDimensions
				{
					ImageDimensions() noexcept: width{0ull}, height{0ull}, depth{0ull} {}
					ImageDimensions(std::size_t width = 0ull, std::size_t height = 0ull, std::size_t depth = 1ull) noexcept : width{width}, height{height}, depth{depth} {}
					ImageDimensions(const ImageDimensions& other) noexcept = default;
					ImageDimensions(ImageDimensions&& other) noexcept = default;
					ImageDimensions& operator=(const ImageDimensions& other) noexcept = default;
					ImageDimensions& operator=(ImageDimensions&& other) noexcept = default;

					std::size_t width;
					std::size_t height;
					std::size_t depth;
				};

				struct HostPitch
				{
					std::size_t row_pitch{0ull};
					std::size_t slice_pitch{0ull};
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
					R = 0,
					G = 1,
					B = 2,
					A = 3
				};

				// [ 8 bit type identifier | 8 bit data type size in bytes ]
				enum class HostDataType : uint16_t
				{
					INT8	= (uint16_t{0} << 8) | uint16_t{1},
					INT16	= (uint16_t{1} << 8) | uint16_t{2},
					INT32	= (uint16_t{2} << 8) | uint16_t{4},
					UINT8	= (uint16_t{3} << 8) | uint16_t{1},
					UINT16	= (uint16_t{4} << 8) | uint16_t{2},
					UINT32	= (uint16_t{5} << 8) | uint16_t{4},
					HALF	= (uint16_t{6} << 8) | uint16_t{2},
					FLOAT	= (uint16_t{7} << 8) | uint16_t{4}
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

				struct ImageOffset
				{
					size_t offset_width{0ull};
					size_t offset_height{0ull};
					size_t offset_depth{0ull};
				};

				struct ImageRegion
				{
					ImageOffset offset;
					ImageDimensions dimensions;
					HostPitch pitch;
				};

				struct HostFormat
				{
					ImageRegion im_region;
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
				// TODO: Support different host channel orders and data types and do automatic conversion.
				inline CLEvent write(const HostFormat& format, const void* data_ptr, bool invalidate = false, ChannelDefaultValue default_value = ChannelDefaultValue::Zeros);
				inline CLEvent read(const HostFormat& format, void* data_ptr, ChannelDefaultValue default_value = ChannelDefaultValue::Zeros);
				// with dependencies
				template<typename DepIterator>
				inline CLEvent write(const HostFormat& format, const void* data_ptr, DepIterator dep_begin, DepIterator dep_end, bool invalidate = false, ChannelDefaultValue default_value = ChannelDefaultValue::Zeros);
				template<typename DepIterator>
				inline CLEvent read(const HostFormat& format, void* data_ptr, DepIterator dep_begin, DepIterator dep_end, ChannelDefaultValue default_value = ChannelDefaultValue::Zeros);

			private:
				CLEvent img_write(const HostFormat& format, const void* data_ptr, bool invalidate = false, ChannelDefaultValue default_value = ChannelDefaultValue::Zeros);
				CLEvent img_read(const HostFormat& format, void* data_ptr, ChannelDefaultValue default_value = ChannelDefaultValue::Zeros);
				bool match_format(const HostFormat& format);

				cl_mem m_image;			
				ImageDesc m_image_desc;
				std::vector<cl_event> m_event_cache;
				std::shared_ptr<CLState> m_cl_state;
			};

			inline CLEvent ocl_template_matching::impl::cl::CLImage::write(const HostFormat& format, const void* data_ptr, bool invalidate, ChannelDefaultValue default_value)
			{
				m_event_cache.clear();
				return img_write(format, data_ptr, invalidate, default_value);
			}

			inline CLEvent ocl_template_matching::impl::cl::CLImage::read(const HostFormat& format, void* data_ptr, ChannelDefaultValue default_value)
			{
				m_event_cache.clear();
				return img_read(format, data_ptr, default_value);
			}

			template<typename DepIterator>
			inline CLEvent ocl_template_matching::impl::cl::CLImage::write(const HostFormat& format, const void* data_ptr, DepIterator dep_begin, DepIterator dep_end, bool invalidate, ChannelDefaultValue default_value)
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

			// global operators
			inline bool operator==(const CLImage::HostChannelOrder& rhs, const CLImage::HostChannelOrder& lhs)
			{
				return ((lhs.num_channels == rhs.num_channels) &&
					(!(lhs.num_channels >= 1) || (lhs.channel_order[0] == rhs.channel_order[0])) &&
					(!(lhs.num_channels >= 2) || (lhs.channel_order[1] == rhs.channel_order[1])) &&
					(!(lhs.num_channels >= 3) || (lhs.channel_order[2] == rhs.channel_order[2])) &&
					(!(lhs.num_channels >= 4) || (lhs.channel_order[3] == rhs.channel_order[3])));
			}
			inline bool operator!=(const CLImage::HostChannelOrder& rhs, const CLImage::HostChannelOrder& lhs) { return !(rhs == lhs); }
		#pragma endregion
		}
	}
}

#endif