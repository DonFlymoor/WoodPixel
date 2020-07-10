/** \file simple_cl.h
*	\author Fabian Friederichs
*
*	\brief Provides a minimal set of C++ wrappers for basic OpenCL 1.2 facilities like programs, kernels, buffers and images.
*
*	The classes Context, Program, Buffer and Image are declared in this header. Context abstracts the creation of an OpenCL context, command queue and so on.
*	Program is able to compile OpenCL-C sources and extract all kernel functions which can then be invoked via a type-safe interface.
*	Buffer and Image allow for simplified creation of buffers and images as well as reading and writing from/to them.
*	Event objects are returned and can be used to synchronize between kernel invokes, write and read operations.
*/

#ifndef _SIMPLE_CL_HPP_
#define _SIMPLE_CL_HPP_

#include <CL/cl.h>
#include <simple_cl_error.hpp>
#include <string>
#include <iostream>
#include <vector>
#include <sstream>
#include <unordered_map>
#include <future>
#include <atomic>
#include <memory>
#include <cstdint>
#include <cassert>
#include <array>

/**
*	\namespace simple_cl
*	\brief Contains all the OpenCL template matching functionality.*
*/
namespace simple_cl
{
	/**
	*	\namespace simple_cl::meta
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
		*	\namespace simple_cl::meta::detail
		*	\brief Encapsulates some implementation detail of the simple_cl::meta namespace
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
		struct conjunction<First, Rest...> : std::conditional<bool(First::value), conjunction<Rest...>, First>::type {};

		/**
			 *	\brief		Evaluates the maximum size of a list of types at compile time.
			 *	\tparam ...  Arbitrary list of types.
			*/
		template <typename ...>
		struct max_size_of;

		/**
		 *	\brief		Evaluates the maximum size of a list of types at compile time.
		 *	\tparam T1  Last type.
		*/
		template <typename T1>
		struct max_size_of<T1>
		{
			static constexpr std::size_t value{sizeof(T1)}; ///<	Maximum size of all types in the type list.
		};

		/**
		 *	\brief		Evaluates the maximum size of a list of types at compile time.
		 *	\tparam	T1	First type.
		 *	\tparam ... Arbitrary list of types.
		*/
		template <typename T1, typename ... T>
		struct max_size_of<T1, T...>
		{
			static constexpr std::size_t value{
				std::conditional<
					(sizeof(T1) > max_size_of<T...>::value),
						std::integral_constant<std::size_t, sizeof(T1)>,
						std::integral_constant<std::size_t, max_size_of<T...>::value>
				>::type::value
			};  ///<	Maximum size of all types in the type list.
		};

		/**
		 *	\brief		Evaluates the maximum alignment of a list of types at compile time.
		 *	\tparam ...  Arbitrary list of types.
		*/
		template <typename ...>
		struct max_align_of;

		/**
		 *	\brief		Evaluates the maximum alignment of a list of types at compile time.
		 *	\tparam T1  Last type.
		*/
		template <typename T1>
		struct max_align_of<T1>
		{
			static constexpr std::size_t value{alignof(T1)}; ///<	Maximum alignment of all types in the type list.
		};

		/**
		 *	\brief		Evaluates the maximum alignment of a list of types at compile time.
		 *	\tparam	T1	First type.
		 *	\tparam ... Arbitrary list of types.
		*/
		template <typename T1, typename ... T>
		struct max_align_of<T1, T...>
		{
			static constexpr std::size_t value{
				std::conditional<
					(alignof(T1) > max_align_of<T...>::value),
						std::integral_constant<std::size_t, alignof(T1)>,
						std::integral_constant<std::size_t, max_align_of<T...>::value>
				>::type::value
			}; ///<	Maximum alignment of all types in the type list.
		};

		/**
		 *	\brief		Strips reference and cv qualification from a type.
		 *	\tparam	T	Type.
		 */
		template <typename T>
		using bare_type_t = typename std::remove_cv<typename std::remove_reference<T>::type>::type;
	}

	
	/**
	*	\namespace simple_cl::util
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

		// memory alignment stuff
		/**
		*	\brief Returns a size greater than or equal to 'size' which is a multiple of 'alignment'.
		*	\tparam	alignment	Desired alignment. Must be a power of 2.
		*	\param	size		Input size.
		*	\return				Size >= 'size' which is a multiple of 'alignment'.		
		*/
		template <std::size_t alignment>
		constexpr size_t calc_aligned_size(std::size_t size)
		{
			return size + ((alignment - (size & (alignment - 1))) & (alignment - 1));
		}

		/**
		*	\brief Returns a size greater than or equal to 'size' which is a multiple of 'alignment'.
		*	\param	alignment	Desired alignment. Must be a power of 2.
		*	\param	size		Input size.
		*	\return				Size >= 'size' which is a multiple of 'alignment'.
		*/
		constexpr std::size_t calc_aligned_size(std::size_t size, std::size_t alignment)
		{
			return size + ((alignment - (size & (alignment - 1))) & (alignment - 1));
		}

		/**
			*	\brief		Returns true if n is a power of 2.
			*	\param n	n
			*	\return		true if n is a power of 2, false otherwise.
		*/
		constexpr bool is_power_of_two(std::size_t n)
		{
			return n && (n & (n - 1)) == 0;
		}

		/**
			*	\brief		Rounds up n to the nearest power of 2.
			*	\param n	n
			*	\return		Nearest power of 2 >= n.
		*/
		inline std::size_t next_power_of_two(std::size_t n)
		{
			if(is_power_of_two(n)) return n;
			std::size_t ct{0};
			while(n != 0)
			{
				n = n >> 1;
				++ct;
			}
			return std::size_t{1ull >> (ct + 1)};
		}
	}

	/**
	*	\namespace simple_cl::cl
	*	\brief Encapsulates implementation of OpenCL wrappers.
	*/
	namespace cl
	{
		namespace constants
		{
			/// Maximum work dim of OpenCL kernels
			constexpr std::size_t OCL_KERNEL_MAX_WORK_DIM{3}; // maximum work dim of opencl kernels
			/// Maximum size of an RGBA fill color
			constexpr std::size_t OCL_MAX_FILL_COLOR_BYTES{4 * sizeof(float)};
			/// Invalid color channel index
			constexpr std::size_t INVALID_COLOR_CHANNEL_INDEX{0xDEADBEEF};
		}

		#pragma region context
		/// Callback function used during OpenCL context creation.
		void create_context_callback(const char* errinfo, const void* private_info, std::size_t cb, void* user_data);

		/**
			*	\brief Creates and manages OpenCL platform, device, context and command queue
			*
			*	This class creates the basic OpenCL state needed to run kernels and create buffers and images.
			*	The constructor is deleted. Please use the factory function createCLInstance(...) instead to retrieve a std::shared_ptr<Context> to an instance of this class.
			*	This way the lifetime of the Context object is ensured to outlive the consuming classes Buffer, Image and so on.
		*/
		class Context
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
				* \brief This factory function creates a new instance of Context and returns a std::shared_ptr<Context> to this instance.
				*
				*	Use this function to create an instance of Context. The other classes all depend on a valid instance. To ensure the instance outlives
				*	created Program, Buffer and Image objects, shared pointers are distributed to these instances.
				* 
				*	\param platform_index	Index of the platform to create the context from.
				*	\param device_index		Index of the device in the selected platform to create the context for.
				*	\return					A shared pointer to the newly created Context instance. Use this for instantiating the other wrapper classes.
			*/
			static std::shared_ptr<Context> createInstance(std::size_t platform_index, std::size_t device_index);

			/// Destructor.
			~Context();

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
				*	\brief	Returns the CLPlatform info struct of the selected platform.
				*	\return  Returns the CLPlatform info struct of the selected platform.
			*/
			const CLPlatform& get_selected_platform() const;

			/**
				*	\brief	Returns the CLDevice info struct of the selected device.
				*	\return  Returns the CLDevice info struct of the selected device.
			*/
			const CLDevice& get_selected_device() const;

			/**
				*	\brief	Prints detailed information about the selected platform.
			*/
			void print_selected_platform_info() const;

			/**
				*	\brief	Prints detailed infomation about the selected device.
			*/
			void print_selected_device_info() const;

			/**
				*	\brief	Prints detailed information about all suitable (OpenCL 1.2+) platforms and devices available on the system.
				*	\param	available_platforms		Vector of CLPlatform's as received from read_platform_and_device_info().
			*/
			static void print_platform_and_device_info(const std::vector<CLPlatform>& available_platforms);

			/**
				*	\brief	Prints detailed information about all suitable (OpenCL 1.2+) platforms and devices available on the system.
			*/
			void print_platform_and_device_info();

			/**
				*	\brief Searches for available platforms and devices and stores suitable ones (OpenCL 1.2+) in the platforms list member.
				*	\return	Returns a vector of CLPlatform's.
			*/
			static std::vector<CLPlatform> read_platform_and_device_info();

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
			Context(std::size_t platform_index, std::size_t device_index);

			/// No copies are allowed.
			Context(const Context&) = delete;
			/// Move the entire state to a new instance.
			Context(Context&& other) noexcept;

			/// No copies are allowed.
			Context& operator=(const Context&) = delete;
			/// Moves the entire state from one instance to another.
			Context& operator=(Context&&) noexcept;

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
			friend std::ostream& operator<<(std::ostream&, const Context::CLPlatform&);
			/// Prints detailed information about the device.
			friend std::ostream& operator<<(std::ostream&, const Context::CLDevice&);
			// opencl callbacks
			/// Callback used while creating the context.
			friend void create_context_callback(const char* errinfo, const void* private_info, std::size_t cb, void* user_data);

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

		#pragma region common
		/**
			* \brief	Specifies whether the kernel can read, write or both. Used for creation of Buffer and Image instances.
		*/
		enum class DeviceAccess : cl_mem_flags
		{
			ReadOnly = CL_MEM_READ_ONLY,		///< Kernel may only read from the created memory object.
			WriteOnly = CL_MEM_WRITE_ONLY,		///< Kernel may only write to the created memory object.
			ReadWrite = CL_MEM_READ_WRITE		///< Kernel may read or write from/to the created memory object.
		};

		/**
			* \brief	Specifies whether the host can read, write or both. Used for creation of Buffer and Image instances.
		*/
		enum class HostAccess : cl_mem_flags
		{
			NoAccess = CL_MEM_HOST_NO_ACCESS,	///< Host cannot read or write the created memory object.
			ReadOnly = CL_MEM_HOST_READ_ONLY,	///< Host may only read from the created memory object.
			WriteOnly = CL_MEM_HOST_WRITE_ONLY,	///< Host may only write to the created memory object.
			ReadWrite = cl_mem_flags{0ull}		///< Host may read or write from/to the created memory object.
		};

		/**
			* \brief	Specifies advanced options regarding usage of a host pointer to initialize or store buffer or image data.
		*/
		enum class HostPointerOption : cl_mem_flags
		{
			None = cl_mem_flags{0ull},				///< Host pointer is ignored.
			AllocHostPtr = CL_MEM_ALLOC_HOST_PTR,	///< Memory for the memory object is allocated in host memory space. Passed host pointer is ignored.
			CopyHostPtr = CL_MEM_COPY_HOST_PTR,		///< Copies data from the given host pointer into the newly created buffer.
			UseHostPtr = CL_MEM_USE_HOST_PTR		///< Memory (pointed to by host pointer) for the buffer was already allocated by the host and is used by OpenCL as data storage.
		};
		#pragma endregion

		#pragma region local memory and samplers

		/**
			*	\brief		Represents some local memory of size sizeof(T) * num_elements. Pass this to a kernel to specify local memory.
			*	\tparam T	Element type in local memory. Should have a well defined size.
		*/
		template <typename T = uint8_t>
		class LocalMemory
		{
		public:
			/**
				*	\brief	Constructs a new LocalMemory instance.
				*	\param num_elements		Desired number of elements in local memory.
			*/
			explicit LocalMemory(std::size_t num_elements = 1ull) : m_num_elements(num_elements) {}
			/// Copy constructor.
			LocalMemory(const LocalMemory&) noexcept = default;
			/// Move constructor.
			LocalMemory(LocalMemory&&) noexcept = default;
			/// Copy assignment operator.
			LocalMemory& operator=(const LocalMemory&) noexcept = default;
			/// Move assignment operator.
			LocalMemory& operator=(LocalMemory&&) noexcept = default;

			/// Used by Program to access argument size.
			std::size_t arg_size() const { return m_num_elements * sizeof(meta::bare_type_t<T>); }
			/// Used by Program to access data pointer (nullptr for local memory!).
			static constexpr const void* arg_data() { return nullptr; }
		private:
			std::size_t m_num_elements; ///< Desired number of elements in local memory.
		};

		/** 
		\namespace simple_cl::cl
		\todo Maybe add sampler objects? 
		*/

		#pragma endregion

		#pragma region program_and_kernels
		// check if a complex type T has member funcions to access data pointer and size (for setting kernel params!)
		/**
		*	\brief Checks if a complex type T is usable as parameter for Program. Negative case.
		*/
		template <typename T, typename = void>
		struct is_cl_param : public std::false_type	{};

		/**
		*	\brief Checks if a complex type T is usable as parameter for Program. Positive case.
		*
		*	Requirements:
		*	1.	The type has to expose a member std::size_t arg_size() (may be const) which returns the size in bytes of the param.
		*	2.	The type has to expose a member const void* arg_data() (may be const) which returns a pointer to arg_size() bytes of data to pass to the kernel as argument.
		*/
		template <typename T>
		struct is_cl_param <T, simple_cl::meta::void_t<
			decltype(std::size_t{std::declval<const T>().arg_size()}), // has const size() member, returning size_t?,
			typename std::enable_if<std::is_convertible<decltype(std::declval<const T>().arg_data()), const void*>::value>::type // has const arg_data() member returning something convertible to const void* ?
		>> : std::true_type {};

		// traits class for handling kernel arguments
		/**
		*	\brief Traits class for convenient processing of kernel arguments. Base template.
		*/
		template <typename T, typename = void>
		struct KernelArgTraits;

		// case: complex type which fulfills requirements of is_cl_param<T>
		/**
		*	\brief Traits class for convenient processing of kernel arguments. T fulfills requirements of is_cl_param<T>.
		*	\tparam T type to check for suitability as a kernel argument.
		*/
		template <typename T>
		struct KernelArgTraits <T, typename std::enable_if<is_cl_param<meta::bare_type_t<T>>::value>::type>
		{
			static std::size_t arg_size(const meta::bare_type_t<T>& arg) { return arg.arg_size(); }
			static const void* arg_data(const meta::bare_type_t<T>& arg) { return static_cast<const void*>(arg.arg_data()); }
		};

		// case: arithmetic type or standard layout type (poc struct, plain array...)
		/**
		*	\brief Traits class for convenient processing of kernel arguments. T is arithmetic type or has standard layout (POC object!) and is not a pointer or nullptr.
		*	\tparam T type to check for suitability as a kernel argument.
		*/
		template <typename T>
		struct KernelArgTraits <T, typename std::enable_if<(std::is_arithmetic<meta::bare_type_t<T>>::value || std::is_standard_layout<meta::bare_type_t<T>>::value) && 
																!std::is_pointer<meta::bare_type_t<T>>::value &&
																!std::is_same<meta::bare_type_t<T>, std::nullptr_t>::value &&
																!is_cl_param<meta::bare_type_t<T>>::value>::type>
		{
			static constexpr std::size_t arg_size(const meta::bare_type_t<T>& arg) { return sizeof(meta::bare_type_t<T>); }
			static const void* arg_data(const meta::bare_type_t<T>& arg) { return static_cast<const void*>(&arg); }
		};

		// general check for allowed argument types. Used to present meaningful error message wenn invoked with wrong types.
		/**
		*	\brief General check for allowed argument types. Used to present meaningful error message wenn invoked with wrong types.
		*
		*	Allowed argument types are types which:
		*	-	fullfill is_cl_param<T> (i.e. expose arg_size and arg_data members!) or
		*	-	arithmetic types like int's and float's or
		*	-	standard layout types (e.g. arithmetic types, POC structs, plain arrays...) and
		*	-	are no pointer types or nullptr_t
		*/
		template <typename T>
		using is_valid_kernel_arg = typename std::conditional<
			(is_cl_param<meta::bare_type_t<T>>::value ||
			std::is_arithmetic<meta::bare_type_t<T>>::value ||
			std::is_standard_layout<meta::bare_type_t<T>>::value) &&
			!std::is_pointer<meta::bare_type_t<T>>::value &&
			!std::is_same<meta::bare_type_t<T>, std::nullptr_t>::value
			, std::true_type, std::false_type>::type;

		/**
		* \brief Handle to some OpenCL event. Can be used to synchronize OpenCL operations.
		*/
		class Event
		{
			friend class Program;
			friend class Buffer;
			friend class Image;

			template<typename DepIterator>
			friend void wait_for_events(DepIterator, DepIterator);

		public:
			/**
			* \brief Constructs a new handle.
			* \param ev OpenCL event to encapsulate.
			*/
			explicit Event(cl_event ev);
			/// Destructor. Internally decreases reference count to the cl_event object.
			~Event();
			/// Copy constructor. Internally increases reference count to the cl_event object.
			Event(const Event& other);
			/// Move constructor.
			Event(Event&& other) noexcept;
			/// Copy assignment. Internally increases reference count to the cl_event object.
			Event& operator=(const Event& other);
			/// Moce assignment.
			Event& operator=(Event&& other) noexcept;

			/**
			* \brief Blocks until the corresponding OpenCL command submitted to the command queue finished execution.
			*/
			void wait() const;
		private:
			/// Used by wait_for_events<T> free function
			static void wait_for_events_(const std::vector<cl_event>& events);
			cl_event m_event; ///< Handled cl_event object.
		};

		/**
		 *	\brief						Waits for a collection of Event's.
		 *	\tparam DependencyIterator	Iterator which refers to a collection of Event's.
		 *	\param begin				Begin of collection.
		 *	\param end					End of collection.
		*/
		template <typename DepIterator>
		inline void wait_for_events(DepIterator begin, DepIterator end)
		{
			static_assert(std::is_same<meta::bare_type_t<typename std::iterator_traits<DepIterator>::value_type>, Event>::value, "[Program]: Dependency iterators must refer to a collection of Event objects.");
			static std::vector<cl_event> event_cache;
			event_cache.clear();
			for(DepIterator it{begin}; it != end; ++it)
				if(it->m_event)
					event_cache.push_back(it->m_event);
			Event::wait_for_events_(event_cache);
		}

		/**
		* \brief Compiles OpenCL-C source code and extracts kernel functions from this source. Found kernels can then be conveniently invoked using the call operator.
		*/
		class Program
		{
		public:
			/// Defines the global and local dimensions of the kernel invocation in terms of dimensions (up to 3) and work items.
			struct ExecParams
			{
				std::size_t work_dim; ///< Dimension of the work groups and the global work volume. Can be 1, 2 or 3.
				std::size_t work_offset[constants::OCL_KERNEL_MAX_WORK_DIM]; ///< Global offset from the origin.
				std::size_t global_work_size[constants::OCL_KERNEL_MAX_WORK_DIM]; ///< Global work volume dimensions.
				std::size_t local_work_size[constants::OCL_KERNEL_MAX_WORK_DIM]; ///< Local work group dimensions.
			};

			/**
			 *	\brief Packs information about the kernel.
			*/
			struct CLKernelInfo
			{
				std::size_t max_work_group_size = 0ull;					///< Maximum number of threads in a work group for this kernel.
				std::size_t local_memory_usage = 0ull;					///< Total local memory usage of this kernel.
				std::size_t private_memory_usage = 0ull;				///< Total private memory usage of this kernel.
				std::size_t preferred_work_group_size_multiple = 0ull;	///< Preferred work group size. Work groups should be a multiple of this size and smaller than max_work_group_size. Total local memory used is another limitation to keep in mind.
			};

			/**
			* \brief Handle to an OpenCL kernel in this program. Useful to circumvent kernel name lookup to improve performance of invokes.
			* \attention This is a non owning handle which becomes invalid if the creating Program instance dies.
			*/
			class CLKernelHandle
			{
			public:
				CLKernelHandle() noexcept = default;
				CLKernelHandle(const CLKernelHandle& other) noexcept = default;
				CLKernelHandle& operator=(const CLKernelHandle& other) noexcept = default;
				~CLKernelHandle() noexcept = default;
				/// Returns information about the kernel.
				const CLKernelInfo& getKernelInfo() const { return m_kernel_info; }
			private:
				friend class Program;
				explicit CLKernelHandle(cl_kernel kernel, const CLKernelInfo& kernel_info) noexcept : m_kernel{kernel}, m_kernel_info{kernel_info} {}
				cl_kernel m_kernel = nullptr;
				CLKernelInfo m_kernel_info;
			};

			/**
			* \brief	Compiles OpenCL-C source code, creates a cl_program object and extracts all the available kernel functions.
			* \param source String containing the entire source code.
			* \param compiler_options String containing compiler options.
			* \param clstate A valid Context intance used to interface with OpenCL.
			*/
			Program(const std::string& source, const std::string& compiler_options, const std::shared_ptr<Context>& clstate);
			/// Destructor. Frees created cl_program and cl_kernel objects.
			~Program();

			// copy / move constructor
			/// Copy construction is not allowed.
			Program(const Program&) = delete;
			/// Moves the entire state into a new instance.
			Program(Program&&) noexcept;

			// copy / move assignment
			/// Copy assignment is not allowed.
			Program& operator=(const Program& other) = delete;
			/// Moves the entire state into another instance.
			Program& operator=(Program&& other) noexcept;

			// no dependencies
			/**
			*	\brief Invokes the kernel 'name' with execution parameters 'exec_params' and passes an arbitrary list of arguments.
			*
			*	All argument types have to satisfy is_valid_kernel_arg<T>.
			*	After submitting the kernel invocation onto the command queue, a Event is returned which can be waited on to achieve blocking behaviour
			*	or passed to other OpenCL wrapper operations to accomplish synchronization with the following operation.
			*
			*	\tparam ...Argtypes	List of argument types.
			*	\param name Name of the kernel function to invoke.
			*	\param exec_params	Defines execution dimensions of global work volume and local work groups for this invocation.
			*	\param args	List of arguments to pass to the kernel.
			*	\return Event object. Calling wait() on this object blocks until the kernel has finished execution.
			*/
			template <typename ... ArgTypes>
			Event operator()(const std::string& name, const ExecParams& exec_params, const ArgTypes&... args)
			{
				static_assert(simple_cl::meta::conjunction<is_valid_kernel_arg<ArgTypes>...>::value, "[Program]: Incompatible kernel argument type.");
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
					throw std::runtime_error("[Program]: Unknown kernel name");
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
			*	After submitting the kernel invocation onto the command queue, a Event is returned which can be waited on to achieve blocking behaviour
			*	or passed to other OpenCL wrapper operations to accomplish synchronization with the following operation.
			*
			*	\tparam ...Argtypes	List of argument types.
			*	\param kernel Handle of the kernel function to invoke.
			*	\param exec_params	Defines execution dimensions of global work volume and local work groups for this invocation.
			*	\param args	List of arguments to pass to the kernel.
			*	\return Event object. Calling wait() on this object blocks until the kernel has finished execution.
			*/
			template <typename ... ArgTypes>
			Event operator()(const CLKernelHandle& kernel, const ExecParams& exec_params, const ArgTypes&... args)
			{
				assert(kernel.m_kernel);
				static_assert(simple_cl::meta::conjunction<is_valid_kernel_arg<ArgTypes>...>::value, "[Program]: Incompatible kernel argument type.");					
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
			*	After submitting the kernel invocation onto the command queue, a Event is returned which can be waited on to achieve blocking behaviour
			*	or passed to other OpenCL wrapper operations to accomplish synchronization with the following operation.
			*
			*	\param name Name of the kernel function to invoke.
			*	\param exec_params	Defines execution dimensions of global work volume and local work groups for this invocation.
			*	\return Event object. Calling wait() on this object blocks until the kernel has finished execution.
			*/
			Event operator()(const std::string& name, const ExecParams& exec_params)
			{
				try
				{
					// invoke kernel
					m_event_cache.clear();
					return invoke(m_kernels.at(name).kernel, m_event_cache, exec_params);
				}
				catch(const std::out_of_range&) // kernel name wasn't found
				{
					throw std::runtime_error("[Program]: Unknown kernel name");
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
			*	After submitting the kernel invocation onto the command queue, a Event is returned which can be waited on to achieve blocking behaviour
			*	or passed to other OpenCL wrapper operations to accomplish synchronization with the following operation.
			*
			*	\param kernel Handle of the kernel function to invoke.
			*	\param exec_params	Defines execution dimensions of global work volume and local work groups for this invocation.
			*	\return Event object. Calling wait() on this object blocks until the kernel has finished execution.
			*/
			Event operator()(const CLKernelHandle& kernel, const ExecParams& exec_params)
			{					
				assert(kernel.m_kernel);
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
			*	After submitting the kernel invocation onto the command queue, a Event is returned which can be waited on to achieve blocking behaviour
			*	or passed to other OpenCL wrapper operations to accomplish synchronization with the following operation.
			*
			*	\tparam ...Argtypes	List of argument types.
			*	\tparam DependencyIterator Iterator which refers to a collection of Event's.
			*	\param name Name of the kernel function to invoke.
			*	\param start_dep_iterator Start iterator of the event collection.
			*	\param end_dep_iterator End iterator of the event collection.
			*	\param exec_params	Defines execution dimensions of global work volume and local work groups for this invocation.
			*	\param args	List of arguments to pass to the kernel.
			*	\return Event object. Calling wait() on this object blocks until the kernel has finished execution.
			*/
			template <typename DependencyIterator, typename ... ArgTypes>
			Event operator()(const std::string& name, DependencyIterator start_dep_iterator, DependencyIterator end_dep_iterator, const ExecParams& exec_params, const ArgTypes&... args)
			{
				static_assert(std::is_same<meta::bare_type_t<typename std::iterator_traits<DependencyIterator>::value_type> , Event>::value , "[Program]: Dependency iterators must refer to a collection of Event objects.");
				static_assert(simple_cl::meta::conjunction<is_valid_kernel_arg<ArgTypes>...>::value, "[Program]: Incompatible kernel argument type.");
				try
				{
					// unpack args
					setKernelArgs < std::size_t{0}, ArgTypes... > (name, args...);

					// invoke kernel
					m_event_cache.clear();
					for(DependencyIterator it{start_dep_iterator}; it != end_dep_iterator; ++it)
						if(it->m_event)
							m_event_cache.push_back(it->m_event);
					return invoke(m_kernels.at(name).kernel, m_event_cache, exec_params);
				}
				catch(const std::out_of_range&) // kernel name wasn't found
				{
					throw std::runtime_error("[Program]: Unknown kernel name");
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
			*	After submitting the kernel invocation onto the command queue, a Event is returned which can be waited on to achieve blocking behaviour
			*	or passed to other OpenCL wrapper operations to accomplish synchronization with the following operation.
			*
			*	\tparam ...Argtypes	List of argument types.
			*	\tparam DependencyIterator Iterator which refers to a collection of Event's.
			*	\param kernel Handle of the kernel function to invoke.
			*	\param start_dep_iterator Start iterator of the event collection.
			*	\param end_dep_iterator End iterator of the event collection.
			*	\param exec_params	Defines execution dimensions of global work volume and local work groups for this invocation.
			*	\param args	List of arguments to pass to the kernel.
			*	\return Event object. Calling wait() on this object blocks until the kernel has finished execution.
			*/
			template <typename DependencyIterator, typename ... ArgTypes>
			Event operator()(const CLKernelHandle& kernel, DependencyIterator start_dep_iterator, DependencyIterator end_dep_iterator, const ExecParams& exec_params, const ArgTypes&... args)
			{
				assert(kernel.m_kernel);
				static_assert(std::is_same<meta::bare_type_t<typename std::iterator_traits<DependencyIterator>::value_type>, Event>::value, "[Program]: Dependency iterators must refer to a collection of Event objects.");
				static_assert(simple_cl::meta::conjunction<is_valid_kernel_arg<ArgTypes>...>::value, "[Program]: Incompatible kernel argument type.");					
				// unpack args
				setKernelArgs < std::size_t{0}, ArgTypes... > (kernel.m_kernel, args...);

				// invoke kernel
				m_event_cache.clear();
				for(DependencyIterator it{start_dep_iterator}; it != end_dep_iterator; ++it)
					if(it->m_event)
						m_event_cache.push_back(it->m_event);
				return invoke(kernel.m_kernel, m_event_cache, exec_params);					
			}

			// overload for zero arguments (with dependencies)
			/**
			*	\brief Invokes the kernel 'name' with execution parameters 'exec_params' after waiting for a collection of CLEvents.
			*
			*	The kernel waits for finalization of the passed events before it proceeds with its own execution.
			*	After submitting the kernel invocation onto the command queue, a Event is returned which can be waited on to achieve blocking behaviour
			*	or passed to other OpenCL wrapper operations to accomplish synchronization with the following operation.
			*
			*	\tparam DependencyIterator Iterator which refers to a collection of Event's.
			*	\param name Name of the kernel function to invoke.
			*	\param start_dep_iterator Start iterator of the event collection.
			*	\param end_dep_iterator End iterator of the event collection.
			*	\param exec_params	Defines execution dimensions of global work volume and local work groups for this invocation.
			*	\return Event object. Calling wait() on this object blocks until the kernel has finished execution.
			*/
			template <typename DependencyIterator>
			Event operator()(const std::string& name, DependencyIterator start_dep_iterator, DependencyIterator end_dep_iterator, const ExecParams& exec_params)
			{
				static_assert(std::is_same<meta::bare_type_t<typename std::iterator_traits<DependencyIterator>::value_type>, Event>::value, "[Program]: Dependency iterators must refer to a collection of Event objects.");
				try
				{
					// invoke kernel
					m_event_cache.clear();
					for(DependencyIterator it{start_dep_iterator}; it != end_dep_iterator; ++it)
						if(it->m_event)
							m_event_cache.push_back(it->m_event);
					return invoke(m_kernels.at(name).kernel, m_event_cache, exec_params);
				}
				catch(const std::out_of_range&) // kernel name wasn't found
				{
					throw std::runtime_error("[Program]: Unknown kernel name");
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
			*	After submitting the kernel invocation onto the command queue, a Event is returned which can be waited on to achieve blocking behaviour
			*	or passed to other OpenCL wrapper operations to accomplish synchronization with the following operation.
			*
			*	\tparam DependencyIterator Iterator which refers to a collection of Event's.
			*	\param kernel Handle of the kernel function to invoke.
			*	\param start_dep_iterator Start iterator of the event collection.
			*	\param end_dep_iterator End iterator of the event collection.
			*	\param exec_params	Defines execution dimensions of global work volume and local work groups for this invocation.
			*	\return Event object. Calling wait() on this object blocks until the kernel has finished execution.
			*/
			template <typename DependencyIterator>
			Event operator()(const CLKernelHandle& kernel, DependencyIterator start_dep_iterator, DependencyIterator end_dep_iterator, const ExecParams& exec_params)
			{				
				assert(kernel.m_kernel);
				static_assert(std::is_same<meta::bare_type_t<typename std::iterator_traits<DependencyIterator>::value_type>, Event>::value, "[Program]: Dependency iterators must refer to a collection of Event objects.");
				// invoke kernel
				m_event_cache.clear();
				for(DependencyIterator it{start_dep_iterator}; it != end_dep_iterator; ++it)
					if(it->m_event)
						m_event_cache.push_back(it->m_event);
				return invoke(kernel.m_kernel, m_event_cache, exec_params);
			}

			// retrieve kernel handle
			/**
			*	\brief Returns a kernel handle to the kernel with name name.
			*	\param name	Name of the kernel to create a handle of.
			*/
			CLKernelHandle getKernel(const std::string& name) const;

			/**
			 *	\brief			Returns information about the kernel, specifically information about preferred work group size and memory usage.
			 *	\param name		Name of the kernel.
			 *	\return			CLKernelInfo struct filled with information about the kernel.
			*/
			CLKernelInfo getKernelInfo(const std::string& name) const;

			/**
			 *	\brief			Returns information about the kernel, specifically information about preferred work group size and memory usage.
			 *	\param kernel	Handle to the kernel.
			 *	\return			CLKernelInfo struct filled with information about the kernel.
			*/
			CLKernelInfo getKernelInfo(const CLKernelHandle& kernel) const;

		private:
			/// Cleans up internal state.
			void cleanup() noexcept;

			/**
				* \brief Holds running id and OpenCL kernel object handle.
			*/
			struct CLKernel
			{
				std::size_t id;		///< Running id
				CLKernelInfo kernel_info; ///< Information about the kernel
				cl_kernel kernel;	///< OpenCL kernel object handle
			};

			// invoke kernel
			/**
			*	\brief invokes the kernel.
			*	\param kernel	OpenCL kernel object handle
			*	\param dep_events	Vector of events to wait for. (std::vector because we need them in contiguous memory for the API call)
			*	\param exec_params	Execution dimensions.
			*/
			Event invoke(cl_kernel kernel, const std::vector<cl_event>& dep_events, const ExecParams& exec_params);
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
				setKernelArgsImpl(name, index, KernelArgTraits<FirstArgType>::arg_size(first_arg), KernelArgTraits<FirstArgType>::arg_data(first_arg));
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
				setKernelArgsImpl(kernel, index, KernelArgTraits<FirstArgType>::arg_size(first_arg), KernelArgTraits<FirstArgType>::arg_data(first_arg));
			}

			std::string m_source;	///< OpenCL program source code.
			std::string m_options;	///< OpenCL-C compiler options string.
			std::unordered_map<std::string, CLKernel> m_kernels;	///< Map of kernels found in the program, keyed by kernel name.
			cl_program m_cl_program;	///< OpenCL program object handle
			std::shared_ptr<Context> m_cl_state;	///< Shared pointer to some valid Context instance.
			std::vector<cl_event> m_event_cache;	///< Used for caching lists of events in contiguous memory.
		};
		#pragma endregion
		
		#pragma region buffers
		/**
			* \brief	Packages all memory creation options for instantiating a Buffer or Image object.
		*/
		struct MemoryFlags
		{
			DeviceAccess device_access;				///< Device access option.
			HostAccess host_access;					///< Host access option.
			HostPointerOption host_pointer_option;	///< Host pointer option.
		};

		/**
			* \brief Encapsulates creation and read / write operations on OpenCL buffer objects.
			* \todo	Optimize read/write with iterators. Exploit contiguous memory using plain memcpy!.
		*/
		class Buffer
		{
		public:

			/**
				*	\brief			Creates a new Buffer instance and allocates an OpenCL buffer.
				*	\param size		Size of the buffer to be allocated in bytes.
				*	\param flags	OpenCL flags for buffer creation.
				*	\param clstate 
				*	\param hostptr 
				*	\return 
			*/
			Buffer(std::size_t size, const MemoryFlags& flags, const std::shared_ptr<Context>& clstate, void* hostptr = nullptr);

			~Buffer() noexcept;
			Buffer(const Buffer&) = delete;
			Buffer(Buffer&& other) noexcept;
			Buffer& operator=(const Buffer&) = delete;
			Buffer& operator=(Buffer&& other) noexcept;

			/** 
			*	\brief Copies data pointed to by data into the OpenCL buffer.
			*	
			*	Copies data pointed to by data into the OpenCL buffer. The function returns a Event which can be waited upon. It refers to the unmap command after copying.
			*	Setting invalidate = true invalidates the written buffer region (all data that was not written is now in undefined state!) but most likely increases performance
			*	due to less synchronization overhead in the driver.
			*
			*	\param[in]		data		Points to the data to be written into the buffer.
			*	\param[in]		length		Length of the data to be written in bytes. If 0 (default), the whole buffer will be written and the offset is ignored.
			*	\param[in]		offset		Offset into the buffer where the region to be written begins. Ignored if length is 0.
			*	\param[in]		invalidate	If true, the written region will be invalidated which provides performance benefits in most cases.
			*
			*	\return			Returns a Event object which can be waited upon either by other OpenCL operations or explicitely to block until the data is synchronized with OpenCL.
			*	
			*	\attention		This is a low level function. Please consider using one of the type-safe versions instead. If this function is used directly make sure that access to data*
			*					in the region [data, data + length - 1] does not produce access violations!
			*/
			inline Event write_bytes(const void* data, std::size_t length = 0ull, std::size_t offset = 0ull, bool invalidate = false);

			/**
			*	Copies data from the OpenCL buffer into the memory region pointed to by data.
			*
			*	Copies data from the OpenCL buffer into the memory region pointed to by data. The function returns a Event which can be waited upon. It refers to the unmap command after copying.
			*
			*	\param[out]		data		Points to the memory region the buffer should be read into.
			*	\param[in]		length		Length of the data to be read in bytes. If 0 (default), the whole buffer will be read and the offset is ignored.
			*	\param[in]		offset		Offset into the buffer where the region to be read begins. Ignored if length is 0.
			*
			*	\return			Returns a Event object which can be waited upon either by other OpenCL operations or explicitely to block until the data is synchronized with OpenCL.
			*
			*	\attention		This is a low level function. Please consider using one of the type-safe versions instead. If this function is used directly make sure that access to data*
			*					in the region [data, data + length - 1] does not produce access violations!
			*/
			inline Event read_bytes(void* data, std::size_t length = 0ull, std::size_t offset = 0ull);

			/** 
			*	Copies data pointed to by data into the OpenCL buffer after waiting on a list of dependencies (Event's).
			*
			*	Copies data pointed to by data into the OpenCL buffer. Before the buffer is mapped for writing, OpenCL waits for the provided Event's.
			*	The function returns a Event which can be waited upon. It refers to the unmap command after copying.
			*	Setting invalidate = true invalidates the written buffer region (all data that was not written is now in undefined state!) but most likely increases performance
			*	due to less synchronization overhead in the driver.
			*
			*	\tparam			DepIterator	Input iterator to iterate over a collection of Event's.
			*
			*	\param[in]		data		Points to the data to be written into the buffer.
			*	\param[in]		dep_begin	Start iterator of a collection of Event's.
			*	\param[in]		dep_end		End iterator of a collection of Event's.
			*	\param[in]		length		Length of the data to be written in bytes. If 0 (default), the whole buffer will be written and the offset is ignored.
			*	\param[in]		offset		Offset into the buffer where the region to be written begins. Ignored if length is 0.
			*	\param[in]		invalidate	If true, the written region will be invalidated which provides performance benefits in most cases.
			*
			*	\return			Returns a Event object which can be waited upon either by other OpenCL operations or explicitely to block until the data is synchronized with OpenCL.
			*
			*	\attention		This is a low level function. Please consider using one of the type-safe versions instead. If this function is used directly make sure that access to data*
			*					in the region [data, data + length - 1] does not produce access violations!
			*/
			template <typename DepIterator>
			inline Event write_bytes(const void* data, DepIterator dep_begin, DepIterator dep_end, std::size_t length = 0ull, std::size_t offset = 0ull, bool invalidate = false);

			/**
			*	Copies data from the OpenCL buffer into the memory region pointed to by data after waiting on a list of dependencies (Event's).
			*
			*	Copies data from the OpenCL buffer into the memory region pointed to by data. Before the buffer is mapped for reading, OpenCL waits for the provided Event's.
			*	The function returns a Event which can be waited upon. It refers to the unmap command after copying.
			*
			*	\tparam			DepIterator	Input iterator to iterate over a collection of Event's.
			*
			*	\param[out]		data		Points to the memory region the buffer should be read into.
			*	\param[in]		dep_begin	Start iterator of a collection of Event's.
			*	\param[in]		dep_end		End iterator of a collection of Event's.
			*	\param[in]		length		Length of the data to be read in bytes. If 0 (default), the whole buffer will be read and the offset is ignored.
			*	\param[in]		offset		Offset into the buffer where the region to be read begins. Ignored if length is 0.
			*
			*	\return			Returns a Event object which can be waited upon either by other OpenCL operations or explicitely to block until the data is synchronized with OpenCL.
			*
			*	\attention		This is a low level function. Please consider using one of the type-safe versions instead. If this function is used directly make sure that access to data*
			*					in the region [data, data + length - 1] does not produce access violations!
			*/
			template <typename DepIterator>
			inline Event read_bytes(void* data, DepIterator dep_begin, DepIterator dep_end, std::size_t length = 0ull, std::size_t offset = 0ull);

			// high level read / write
			/**
			*	\brief Writes some collection of POC data into the buffer, starting at some byte offset.
			*	\tparam		DataIterator	Some iterator type fulfilling the LegacyInputIterator named requirement and referring to POC data (std::is_standard_layout<T>).
			*	\param		data_begin		Begin iterator of data.
			*	\param		data_end		End iterator of data.
			*	\param		offset			Offset into the OpenCL buffer: offset * sizeof(std::iterator_traits<DataIterator>::value_type) bytes.
			*	\param		invalidate		When true, invalidates the whole mapped memory region. This increases transfer performance in most cases.
			*	\return		Returns a Event object which can be waited upon either by other OpenCL operations or explicitely to block until the data is synchronized with OpenCL.
			*/
			template <typename DataIterator>
			inline Event write(DataIterator data_begin, DataIterator data_end, std::size_t offset = 0ull, bool invalidate = false);

			/**
			*	\brief Reads some collection of POC data from the buffer, starting at some byte offset.
			*	\tparam		DataIterator	Some iterator type fulfilling the LegacyOutputIterator named requirement and referring to POC data (std::is_standard_layout<T>).
			*	\param		data_begin		Begin iterator of data.
			*	\param		num_elements	Number of elements to read from the OpenCL buffer.
			*	\param		offset			Offset into the OpenCL buffer: offset * sizeof(std::iterator_traits<DataIterator>::value_type) bytes.
			*	\return		Returns a Event object which can be waited upon either by other OpenCL operations or explicitely to block until the data is synchronized with OpenCL.
			*/
			template <typename DataIterator>
			inline Event read(DataIterator data_begin, std::size_t num_elements, std::size_t offset = 0ull);

			// with dependencies
			/**
			*	\brief Writes some collection of POC data into the buffer, starting at some byte offset, after waiting on a list of Event's.
			*	\tparam		DataIterator	Some iterator type fulfilling the LegacyInputIterator named requirement and referring to POC data (std::is_standard_layout<T>).
			*	\tparam		DepIterator		Some iterator type fulfilling the LegacyInputIterator named requirement and referring to Event objects.
			*	\param		data_begin		Begin iterator of data.
			*	\param		data_end		End iterator of data.
			*	\param		dep_begin		Begin iterator of Event collection.
			*	\param		dep_end			End iterator of Event collection.
			*	\param		offset			Offset into the OpenCL buffer: offset * sizeof(std::iterator_traits<DataIterator>::value_type) bytes.
			*	\param		invalidate		When true, invalidates the whole mapped memory region. This increases transfer performance in most cases.
			*	\return		Returns a Event object which can be waited upon either by other OpenCL operations or explicitely to block until the data is synchronized with OpenCL.
			*/
			template <typename DataIterator, typename DepIterator>
			inline Event write(DataIterator data_begin, DataIterator data_end, DepIterator dep_begin, DepIterator dep_end, std::size_t offset = 0ull, bool invalidate = false);

			/**
			*	\brief Reads some collection of POC data from the buffer, starting at some byte offset, after waiting on a list of Event's.
			*	\tparam		DataIterator	Some iterator type fulfilling the LegacyOutputIterator named requirement and referring to POC data (std::is_standard_layout<T>).
			*	\tparam		DepIterator		Some iterator type fulfilling the LegacyInputIterator named requirement and referring to Event objects.
			*	\param		data_begin		Begin iterator of data.
			*	\param		num_elements	Number of elements to read from the OpenCL buffer.
			*	\param		dep_begin		Begin iterator of Event collection.
			*	\param		dep_end			End iterator of Event collection.
			*	\param		offset			Offset into the OpenCL buffer: offset * sizeof(std::iterator_traits<DataIterator>::value_type) bytes.
			*	\return		Returns a Event object which can be waited upon either by other OpenCL operations or explicitely to block until the data is synchronized with OpenCL.
			*/
			template <typename DataIterator, typename DepIterator>
			inline Event read(DataIterator data_begin, std::size_t num_elements, DepIterator dep_begin, DepIterator dep_end, std::size_t offset = 0ull);

			/// Reports size of allocated device memory in bytes.
			std::size_t size() const noexcept;

			/** 
			*	\brief Used for interfacing with Program (this class can be used as kernel argument)
			*	\return	Returns size of a cl_mem handle.
			*/
			static constexpr std::size_t arg_size() { return sizeof(cl_mem); }
			/**
			*	\brief Used for interfacing with Program (this class can be used as kernel argument)
			*	\return	Returns pointer to the cl_mem handle.
			*/
			const void* arg_data() const { return &m_cl_memory; }
				
		private:
				
			/**
				*	\brief	Writes some raw data into the OpenCL buffer.
				*	\param data			Data pointer.
				*	\param length		Length of data in bytes.
				*	\param offset		Offset into the buffer in bytes.
				*	\param invalidate	If true, the mapped region is invalidated before writing.
				*	\return	Returns a Event of the unmap operation.
			*/
			Event buf_write(const void* data, std::size_t length = 0ull, std::size_t offset = 0ull, bool invalidate = false);
				
			/**
				*	\brief	Reads some raw data from the OpenCL buffer.
				*	\param[out] data			Data pointer.
				*	\param length				Length of data to read in bytes.
				*	\param offset				Offset into the buffer in bytes.
				*	\return						Returns a Event of the unmap operation.
			*/
			Event buf_read(void* data, std::size_t length = 0ull, std::size_t offset = 0ull) const;

			/**
				*	\brief	Maps the memory region specified by length and offset into the host's address space.
				*	\param length		Length of the region to be mapped in bytes.
				*	\param offset		Offset into the buffer in bytes.
				*	\param write		If true, the region is mapped for write access.
				*	\param invalidate	Invalidates the buffer region in case of write access. Ignored if write is false.
				*	\return				Returns a pointer to the mapped memory region. Reading from that region is undefined if write is true, writing is undefined otherwise.
			*/
			void* map_buffer(std::size_t length, std::size_t offset, bool write, bool invalidate = false);

			/**
				*	\brief	Unmaps a buffer region mapped previously.
				*	\param bufptr	Pointer to the beginning (!) of the memory region to be unmapped.
				*	\return			Event of the unmap operation. Blocking behaviour can be achieved if wait is called immediately, e.g.: unmap_buffer(ptr).wait();
			*/
			Event unmap_buffer(void* bufptr);

			cl_mem m_cl_memory;	///< Handle to allocated OpenCL buffer.
			MemoryFlags m_flags;						///< Memory flags used to create the buffer.
			void* m_hostptr;							///< Host pointer used to create the buffer.
			std::size_t m_size;							///< Size in bytes of the allocated buffer memory.
			std::shared_ptr<Context> m_cl_state;		///< Shared pointer to a valid Context instance.
			std::vector<cl_event> m_event_cache;		///< Used for caching cl_event's in contiguous memory before calling the OpenCL API functions.
		};

		Event simple_cl::cl::Buffer::write_bytes(const void* data, std::size_t length, std::size_t offset, bool invalidate)
		{
			m_event_cache.clear();
			return buf_write(data, length, offset, invalidate);
		}

		Event simple_cl::cl::Buffer::read_bytes(void* data, std::size_t length, std::size_t offset)
		{
			m_event_cache.clear();
			return buf_read(data, length, offset);
		}

		template<typename DepIterator>
		inline Event simple_cl::cl::Buffer::write_bytes(const void* data, DepIterator dep_begin, DepIterator dep_end, std::size_t length, std::size_t offset, bool invalidate)
		{
			static_assert(std::is_same<meta::bare_type_t<typename std::iterator_traits<DepIterator>::value_type>, Event>::value, "[Image]: Dependency iterators must refer to a collection of Event objects.");
			m_event_cache.clear();
			for(DepIterator it{dep_begin}; it != dep_end; ++it)
				if(it->m_event)
					m_event_cache.push_back(it->m_event);
			return buf_write(data, length, offset, invalidate);
		}

		template<typename DepIterator>
		inline Event simple_cl::cl::Buffer::read_bytes(void* data, DepIterator dep_begin, DepIterator dep_end, std::size_t length, std::size_t offset)
		{
			static_assert(std::is_same<meta::bare_type_t<typename std::iterator_traits<DepIterator>::value_type>, Event>::value, "[Image]: Dependency iterators must refer to a collection of Event objects.");
			m_event_cache.clear();
			for(DepIterator it{dep_begin}; it != dep_end; ++it)
				if(it->m_event)
					m_event_cache.push_back(it->m_event);
			return buf_read(data, length, offset);
		}

		template<typename DataIterator>
		inline Event simple_cl::cl::Buffer::write(DataIterator data_begin, DataIterator data_end, std::size_t offset, bool invalidate)
		{
			if(m_flags.host_access == HostAccess::ReadOnly || m_flags.host_access == HostAccess::NoAccess)
				throw std::runtime_error("[Buffer]: Writing to a read only buffer is not allowed.");
			using elem_t = typename std::iterator_traits<DataIterator>::value_type;
			static_assert(std::is_standard_layout<elem_t>::value, "[Buffer]: Types read and written from and to OpenCL buffers must have standard layout.");
			std::size_t datasize = static_cast<std::size_t>(std::distance(data_begin, data_end)) * sizeof(elem_t);
			std::size_t bufoffset = offset * sizeof(elem_t);
			if(bufoffset + datasize > m_size)
				throw std::out_of_range("[Buffer]: Buffer write failed. Input offset + length out of range.");
			m_event_cache.clear();
			elem_t* bufptr = static_cast<elem_t*>(map_buffer(datasize, bufoffset, true, invalidate));
			std::size_t bufidx = 0;
			for(DataIterator it{data_begin}; it != data_end; ++it)
				bufptr[bufidx++] = *it;
			return unmap_buffer(static_cast<void*>(bufptr));
		}

		template<typename DataIterator>
		inline Event simple_cl::cl::Buffer::read(DataIterator data_begin, std::size_t num_elements, std::size_t offset)
		{
			if(m_flags.host_access == HostAccess::WriteOnly || m_flags.host_access == HostAccess::NoAccess)
				throw std::runtime_error("[Buffer]: Reading from a write only buffer is not allowed.");
			using elem_t = typename std::iterator_traits<DataIterator>::value_type;
			static_assert(std::is_standard_layout<elem_t>::value, "[Buffer]: Types read and written from and to OpenCL buffers must have standard layout.");
			std::size_t datasize = num_elements * sizeof(elem_t);
			std::size_t bufoffset = offset * sizeof(elem_t);
			if(bufoffset + datasize > m_size)
				throw std::out_of_range("[Buffer]: Buffer read failed. Input offset + length out of range.");
			m_event_cache.clear();
			elem_t* bufptr = static_cast<elem_t*>(map_buffer(datasize, bufoffset, false, false));
			DataIterator it = data_begin;
			for(std::size_t i{0ull}; i < num_elements; ++i)
				*(it++) = bufptr[i];
			return unmap_buffer(static_cast<void*>(bufptr));
		}

		template<typename DataIterator, typename DepIterator>
		inline Event simple_cl::cl::Buffer::write(DataIterator data_begin, DataIterator data_end, DepIterator dep_begin, DepIterator dep_end, std::size_t offset, bool invalidate)
		{
			if(m_flags.host_access == HostAccess::ReadOnly || m_flags.host_access == HostAccess::NoAccess)
				throw std::runtime_error("[Buffer]: Writing to a read only buffer is not allowed.");
			static_assert(std::is_same<meta::bare_type_t<typename std::iterator_traits<DepIterator>::value_type>, Event>::value, "[Image]: Dependency iterators must refer to a collection of Event objects.");
			m_event_cache.clear();
			for(DepIterator it{dep_begin}; it != dep_end; ++it)
				if(it->m_event)
					m_event_cache.push_back(it->m_event);
			using elem_t = typename std::iterator_traits<DataIterator>::value_type;
			static_assert(std::is_standard_layout<elem_t>::value, "[Buffer]: Types read and written from and to OpenCL buffers must have standard layout.");
			std::size_t datasize = static_cast<std::size_t>(std::distance(data_begin, data_end)) * sizeof(elem_t);
			std::size_t bufoffset = offset * sizeof(elem_t);
			if(bufoffset + datasize > m_size)
				throw std::out_of_range("[Buffer]: Buffer write failed. Input offset + length out of range.");
			elem_t* bufptr = static_cast<elem_t*>(map_buffer(datasize, bufoffset, true, invalidate));
			std::size_t bufidx = 0;
			for(DataIterator it{data_begin}; it != data_end; ++it)
				bufptr[bufidx++] = *it;
			return unmap_buffer(static_cast<void*>(bufptr));
		}

		template<typename DataIterator, typename DepIterator>
		inline Event simple_cl::cl::Buffer::read(DataIterator data_begin, std::size_t num_elements, DepIterator dep_begin, DepIterator dep_end, std::size_t offset)
		{
			if(m_flags.host_access == HostAccess::WriteOnly || m_flags.host_access == HostAccess::NoAccess)
				throw std::runtime_error("[Buffer]: Reading from a write only buffer is not allowed.");
			static_assert(std::is_same<meta::bare_type_t<typename std::iterator_traits<DepIterator>::value_type>, Event>::value, "[Image]: Dependency iterators must refer to a collection of Event objects.");
			m_event_cache.clear();
			for(DepIterator it{dep_begin}; it != dep_end; ++it)
				if(it->m_event)
					m_event_cache.push_back(it->m_event);
			using elem_t = typename std::iterator_traits<DataIterator>::value_type;
			static_assert(std::is_standard_layout<elem_t>::value, "[Buffer]: Types read and written from and to OpenCL buffers must have standard layout.");
			std::size_t datasize = num_elements * sizeof(elem_t);
			std::size_t bufoffset = offset * sizeof(elem_t);
			if(bufoffset + datasize > m_size)
				throw std::out_of_range("[Buffer]: Buffer read failed. Input offset + length out of range.");
			elem_t* bufptr = static_cast<elem_t*>(map_buffer(datasize, bufoffset, false, false));
			DataIterator it = data_begin;
			for(std::size_t i{0ull}; i < num_elements; ++i)
				*(it++) = bufptr[i];
			return unmap_buffer(static_cast<void*>(bufptr));
		}

		#pragma endregion
			
		#pragma region images

		// TODO: Implement reading and writing with non-matching host vs. image channel order and data type
		/**
		*	\brief	Creates and manages OpenCL image objects and provides basic read and write access.
		*	\todo	Implement reading and writing with non-matching host vs. image channel order and data types.
		*/
		class Image
		{
		public:
			/**
				*	\brief Specifies base type category of channel content. 
			*/
			enum class ChannelBaseType : uint8_t
			{
				Int = uint8_t{0},
				UInt = uint8_t{1},
				Float = uint8_t{2}
			};

			/**
			*	\brief Identifies a color channel.
			*/
			enum class ColorChannel : uint8_t
			{
				R = uint8_t{0},
				G = uint8_t{1},
				B = uint8_t{2},
				A = uint8_t{3}
			};

			/**
				*	\brief	Specifies the type of image object being created. Buffer images are not
				*	\note	Buffer images are not supported yet.
			*/
			enum class ImageType : cl_mem_object_type
			{
				Image1D = CL_MEM_OBJECT_IMAGE1D,			///< 1D image
				Image2D = CL_MEM_OBJECT_IMAGE2D,			///< 2D image
				Image3D = CL_MEM_OBJECT_IMAGE3D,			///< 3D image
				Image1DArray = CL_MEM_OBJECT_IMAGE1D_ARRAY,	///< 1D image array
				Image2DArray = CL_MEM_OBJECT_IMAGE2D_ARRAY	///< 2D image array
			};

			/**
			*	\brief Specifies the number and order of components of the image.
			*
			*	These five formats are the minimal set of required formats for OpenCL 1.2 compliant devices.
			*
			*	\note	This enum encodes additional information in the less significant bits:
			*			[ 32 bits CL constant | 8 bits channel count | 4 bits first channel | 4 bits second channel | 4 bits third channel | 4 bits fourth channel | 8 bits unused ]
			*			This way we can avoid some bulky switch-cases in the implementation.
			*/
			enum class ImageChannelOrder : uint64_t
			{
				R		= (uint64_t{CL_R} << 32)		| (uint64_t{1} << 24) | (uint64_t(static_cast<uint8_t>(ColorChannel::R)) << 20) | (uint64_t(static_cast<uint8_t>(ColorChannel::R)) << 16) | (uint64_t(static_cast<uint8_t>(ColorChannel::R)) << 12) | (uint64_t(static_cast<uint8_t>(ColorChannel::R)) << 8),
				RG		= (uint64_t{CL_RG} << 32)		| (uint64_t{2} << 24) | (uint64_t(static_cast<uint8_t>(ColorChannel::R)) << 20) | (uint64_t(static_cast<uint8_t>(ColorChannel::G)) << 16) | (uint64_t(static_cast<uint8_t>(ColorChannel::G)) << 12) | (uint64_t(static_cast<uint8_t>(ColorChannel::G)) << 8),
				RGBA	= (uint64_t{CL_RGBA} << 32)		| (uint64_t{4} << 24) | (uint64_t(static_cast<uint8_t>(ColorChannel::R)) << 20) | (uint64_t(static_cast<uint8_t>(ColorChannel::G)) << 16) | (uint64_t(static_cast<uint8_t>(ColorChannel::B)) << 12) | (uint64_t(static_cast<uint8_t>(ColorChannel::A)) << 8),
				BGRA	= (uint64_t{CL_BGRA} << 32)		| (uint64_t{4} << 24) | (uint64_t(static_cast<uint8_t>(ColorChannel::B)) << 20) | (uint64_t(static_cast<uint8_t>(ColorChannel::G)) << 16) | (uint64_t(static_cast<uint8_t>(ColorChannel::R)) << 12) | (uint64_t(static_cast<uint8_t>(ColorChannel::A)) << 8),
				sRGBA	= (uint64_t{CL_sRGBA} << 32)	| (uint64_t{4} << 24) | (uint64_t(static_cast<uint8_t>(ColorChannel::R)) << 20) | (uint64_t(static_cast<uint8_t>(ColorChannel::G)) << 16) | (uint64_t(static_cast<uint8_t>(ColorChannel::B)) << 12) | (uint64_t(static_cast<uint8_t>(ColorChannel::A)) << 8)
			};

			/**
			*	\brief	Specifies the channel data type of the image.
			*
			*	These 12 data types are the minimal set of required data types for OpenCL 1.2 compliant devices.
			*	For allowed combinations with different channel orders please see https://www.khronos.org/registry/OpenCL/specs/2.2/html/OpenCL_API.html#image-format-descriptor.
			*	\note	This enum additionally encodes the size in bytes of the data type and the type category in the less significant bits:
			*			[ 32 bit CL constant | 16 bit data type size in bytes | 8 bit base type identifier | 8 bit normalized flag (in case of SNORM or UNORM types!) ]
			*/
			enum class ImageChannelType : uint64_t
			{
				SNORM_INT8		= (uint64_t{CL_SNORM_INT8} << 32)		| (uint64_t{1} << 16) | (uint64_t(static_cast<uint8_t>(ChannelBaseType::Int))	<< 8) | uint64_t{1},
				SNORM_INT16		= (uint64_t{CL_SNORM_INT16} << 32)		| (uint64_t{2} << 16) | (uint64_t(static_cast<uint8_t>(ChannelBaseType::Int))	<< 8) | uint64_t{1},
				UNORM_INT8		= (uint64_t{CL_UNORM_INT8} << 32)		| (uint64_t{1} << 16) | (uint64_t(static_cast<uint8_t>(ChannelBaseType::UInt))	<< 8) | uint64_t{1},
				UNORM_INT16		= (uint64_t{CL_UNORM_INT16} << 32)		| (uint64_t{2} << 16) | (uint64_t(static_cast<uint8_t>(ChannelBaseType::UInt))	<< 8) | uint64_t{1},
				INT8			= (uint64_t{CL_SIGNED_INT8} << 32)		| (uint64_t{1} << 16) | (uint64_t(static_cast<uint8_t>(ChannelBaseType::Int))	<< 8) | uint64_t{0},
				INT16			= (uint64_t{CL_SIGNED_INT16} << 32)		| (uint64_t{2} << 16) | (uint64_t(static_cast<uint8_t>(ChannelBaseType::Int))	<< 8) | uint64_t{0},
				INT32			= (uint64_t{CL_SIGNED_INT32} << 32)		| (uint64_t{4} << 16) | (uint64_t(static_cast<uint8_t>(ChannelBaseType::Int))	<< 8) | uint64_t{0},
				UINT8			= (uint64_t{CL_UNSIGNED_INT8} << 32)	| (uint64_t{1} << 16) | (uint64_t(static_cast<uint8_t>(ChannelBaseType::UInt))	<< 8) | uint64_t{0},
				UINT16			= (uint64_t{CL_UNSIGNED_INT16} << 32)	| (uint64_t{2} << 16) | (uint64_t(static_cast<uint8_t>(ChannelBaseType::UInt))	<< 8) | uint64_t{0},
				UINT32			= (uint64_t{CL_UNSIGNED_INT32} << 32)	| (uint64_t{4} << 16) | (uint64_t(static_cast<uint8_t>(ChannelBaseType::UInt))	<< 8) | uint64_t{0},
				HALF			= (uint64_t{CL_HALF_FLOAT} << 32)		| (uint64_t{2} << 16) | (uint64_t(static_cast<uint8_t>(ChannelBaseType::Float))	<< 8) | uint64_t{0},
				FLOAT			= (uint64_t{CL_FLOAT} << 32)			| (uint64_t{4} << 16) | (uint64_t(static_cast<uint8_t>(ChannelBaseType::Float))	<< 8) | uint64_t{0}
			};

			/**
				*	\brief	Specifies dimensions of an image.
				*
				*	1D images: width = width, height = 1, depth = 1
				*	2D images: width = width, height = height, depth = 1
				*	3D images: width = width, height = height, depth = depth
				*	1D image arrays: width = width, height = #layers, depth = 1
				*	2D image arrays: width = width, height = height, depth = #layers
			*/
			struct ImageDimensions
			{
				ImageDimensions() noexcept: width{0ull}, height{0ull}, depth{0ull} {}
				ImageDimensions(std::size_t width = 0ull, std::size_t height = 1ull, std::size_t depth = 1ull) noexcept : width{width}, height{height}, depth{depth} {}
				ImageDimensions(const ImageDimensions& other) noexcept = default;
				ImageDimensions(ImageDimensions&& other) noexcept = default;
				ImageDimensions& operator=(const ImageDimensions& other) noexcept = default;
				ImageDimensions& operator=(ImageDimensions&& other) noexcept = default;

				std::size_t width;	///< Image width
				std::size_t height;	///< Image height
				std::size_t depth;	///< Image depth
			};

			/**
				* \brief	Specifies the pitch in bytes of rows and slices of the host image (for reading from or writing to).
			*/
			struct HostPitch
			{
				std::size_t row_pitch{0ull};	///< Pitch (length) of one row of the host image, may only be larger than its pixel width.
				std::size_t slice_pitch{0ull};	///< Pitch (length) of one slice (or layer in case of an image array) of the host image, may only be larger than height * row_pitch.
			};

			/**
				*	\brief	Specifies all information for the creation of a new image.
			*/
			struct ImageDesc
			{
				ImageType type;						///< Type of the image, e.g. 1D, 2D, 3D, 1D array or 2D array.
				ImageDimensions dimensions;			///< Dimensions of the image.
				ImageChannelOrder channel_order;	///< Channel order of the image.
				ImageChannelType channel_type;		///< Channel data type.
				MemoryFlags flags;					///< Flags for image creation. Specifies kernel and host access permissions as well as the usage of a host pointer.
				HostPitch pitch;					///< If the host pointer is used for initializing or storing the image, specifies pitch values of the pointed to data.
				void* host_ptr;						///< Pointer to existing host memory for storing an image or initializing the new image. Ignored if MemoryFlags is not one of MemoryFlags::UseHostPtr or MemoryFlags::CopyHostPtr.
			};

			/**
			*	\brief Specifies the data type of a host image.
			*
			*	\note	This enum additionally encodes the size in bytes of the data type and the type category in the less significant bits:
			*			[ 8 bit type identifier | 4 bit data type size in bytes | 4 bit base type identifier ]
			*/
			enum class HostDataType : uint16_t
			{
				INT8	= (uint16_t{0} << 8) | (uint16_t{1} << 4) | (uint16_t(static_cast<uint8_t>(ChannelBaseType::Int))),
				INT16	= (uint16_t{1} << 8) | (uint16_t{2} << 4) | (uint16_t(static_cast<uint8_t>(ChannelBaseType::Int))),
				INT32	= (uint16_t{2} << 8) | (uint16_t{4} << 4) | (uint16_t(static_cast<uint8_t>(ChannelBaseType::Int))),
				UINT8	= (uint16_t{3} << 8) | (uint16_t{1} << 4) | (uint16_t(static_cast<uint8_t>(ChannelBaseType::UInt))),
				UINT16	= (uint16_t{4} << 8) | (uint16_t{2} << 4) | (uint16_t(static_cast<uint8_t>(ChannelBaseType::UInt))),
				UINT32	= (uint16_t{5} << 8) | (uint16_t{4} << 4) | (uint16_t(static_cast<uint8_t>(ChannelBaseType::UInt))),
				HALF	= (uint16_t{6} << 8) | (uint16_t{2} << 4) | (uint16_t(static_cast<uint8_t>(ChannelBaseType::Float))),
				FLOAT	= (uint16_t{7} << 8) | (uint16_t{4} << 4) | (uint16_t(static_cast<uint8_t>(ChannelBaseType::Float)))
			};

			/**
				*	\brief	Specifies the default value read or written if the channel order does not match between host and device.
				*	\note	This is currently ignored until the auto conversion feature is implemented.
			*/
			enum class ChannelDefaultValue : uint8_t
			{
				Zeros,
				Ones
			};

			/**
				*	\brief	Defines the number and order of color channels of a host image. 
				*	\todo	Add constructor overloads for convenient instantiation.
				*	\todo	Provide some constexpr pre defined channel orders which are most common.
			*/
			struct HostChannelOrder
			{
				std::size_t num_channels;
				ColorChannel channels[4];
			};

			/**
				*	\brief	Specifies an offset into the image. Default offsets are 0.
			*/
			struct ImageOffset
			{
				size_t offset_width{0ull};
				size_t offset_height{0ull};
				size_t offset_depth{0ull};
			};

			/**
				*	\brief Specifies an image region for reading or writing. A region consists of an offset and the dimensions of the desired region.
			*/
			struct ImageRegion
			{
				ImageOffset offset;			///< Offset the region into the image.
				ImageDimensions dimensions;	///< Dimensions of the region.					
			};

			/**
				*	\brief Specifies the format of a host image. 
			*/
			struct HostFormat
			{
				HostChannelOrder channel_order;	///< Channel count and order of the host image.
				HostDataType channel_type;		///< Data type of the host image channels.
				HostPitch pitch;				///< Row and slice pitch of the host image.
			};

			/// Returns image channel data type size in bytes.
			static inline std::size_t get_image_channel_type_size(const ImageChannelType type);
			/// Returns host channel data type size in bytes.
			static inline std::size_t get_host_channel_type_size(const HostDataType type);
			/// Returns number of channels in an ImageChanneOrder.
			static inline std::size_t get_num_image_pixel_components(const ImageChannelOrder channel_order);
			/// Returns number of channels in an HostChannelOrder.
			static inline std::size_t get_num_host_pixel_components(const HostChannelOrder& channel_order);
			/// Returns the corresponding OpenCL constant for the specified channel order.
			static inline cl_uint get_image_channel_order_specifier(const ImageChannelOrder channel_order);
			/// Returns the corresponding OpenCL constant for the specified channel data type.
			static inline cl_uint get_image_channel_type_specifier(const ImageChannelType channel_type);
			/// Returns the image channel data type's base type (Int, Uint or Float).
			static inline ChannelBaseType get_image_channel_base_type(const ImageChannelType channel_type);
			/// Returns the host channel data type's base type (Int, Uint or Float).
			static inline ChannelBaseType get_host_channel_base_type(const HostDataType channel_type);
			/// Returns the channel identifier (R, G, B or A) of the image color channel with index 'index'.
			static inline ColorChannel get_image_color_channel(const ImageChannelOrder channel_order, std::size_t index);
			/// Returns if the image channel data type is a normalized integer type
			static inline bool is_image_channel_format_normalized_integer(const ImageChannelType channel_type);
			/// Returns the component index of the color channel. If the desired channel cannot be found, 3 is returned.
			static inline std::size_t get_image_color_channel_index(const ImageChannelOrder channel_order, const ColorChannel channel);

			/**
				*	\brief	Creates a new OpenCL image.
				*	\param clstate		Shared pointer to some valid Context instance.
				*	\param image_desc	Description of image format, access permissions and so on.
			*/
			Image(const std::shared_ptr<Context>& clstate, const ImageDesc& image_desc);
			/// Frees acquired OpenCL resources.
			~Image() noexcept;
			/// Copies are not allowed.
			Image(const Image&) = delete;
			/// Moves the entire state to a new instance (and invalidates other).
			Image(Image&& other) noexcept;
			/// Copy assignment is not allowed.
			Image& operator=(const Image&) = delete;
			/// Moves the entire state to another instance (and invalidates other).
			Image& operator=(Image&& other) noexcept;

			/**
				*	\brief	Reports the width of the image in pixels.
				*	\return Width of the image in pixels.
			*/
			std::size_t width() const;
			/**
				*	\brief	Reports the height of the image in pixels.
				*	\return Height of the image in pixels.
			*/
			std::size_t height() const;
			/**
				*	\brief	Reports the depth of the image in pixels.
				*	\return Depth of the image in pixels.
			*/
			std::size_t depth() const;
			/**
				*	\brief	Reports number of layers of the image in the case the image is of type 1D array or 2D array.
				*	\return Number of layers.
			*/
			std::size_t layers() const;

			// read / write functions
			/**
			*	\brief	Writes data into the image.
			*
			*	Writes image data from data_ptr into the image region defined by img_region. Currently channel base data type (bitness and type (uint, int or float)) and channel order of host and image
			*	must match, otherwise an exception is thrown.
			*	
			*	\param	img_region		Image region (offset and dimensions) of the target image to be written.
			*	\param	format			Host channel data type, channel order and pitch of the host memory region wher the data is read from.
			*	\param	data_ptr		Pointer to imagedata that shall be written into the specified image region.
			*	\param		blocking		If true, this function blocks until the operation is finished. Otherwise, it returns immediately.
			*	\attention					Make sure data_ptr stays valid until the operation is finished when blocking is false! Otherwise this may cause access violations.
			*	\param	default_value	Currently ignored.
			*
			*	\return					Returns a Event object which can be waited upon either by other OpenCL operations or explicitely to block until the data is synchronized with OpenCL.
			*
			*	\note default_value is currently ignored until the automatic conversion feature is implemented.
			*/
			inline Event write(const ImageRegion& img_region, const HostFormat& format, const void* data_ptr, bool blocking = true, ChannelDefaultValue default_value = ChannelDefaultValue::Zeros);

			/**
			*	\brief	Reads data from the image.
			*
			*	Reads image data from the image region defined by img_region and stores it into the the memory region pointed by data_ptr.
			*	Currently channel base data type (bitness and type (uint, int or float)) and channel order of host and image
			*	must match, otherwise an exception is thrown.
			*
			*	\param		img_region		Image region (offset and dimensions) of the target image to be read.
			*	\param		format			Host channel data type, channel order and pitch of the host memory region where the data should be written to.
			*	\param[out]	data_ptr		Pointer to imagedata that shall be written.
			*	\param		blocking		If true, this function blocks until the operation is finished. Otherwise, it returns immediately.
			*	\attention					Make sure data_ptr stays valid until the operation is finished when blocking is false! Otherwise this may cause access violations.
			*	\param		default_value	Currently ignored.
			*
			*	\return					Returns a Event object which can be waited upon either by other OpenCL operations or explicitely to block until the data is synchronized with OpenCL.
			*
			*	\note default_value is currently ignored until the automatic conversion feature is implemented.
			*/
			inline Event read(const ImageRegion& img_region, const HostFormat& format, void* data_ptr, bool blocking = true, ChannelDefaultValue default_value = ChannelDefaultValue::Zeros);
				
			/**
			*	\brief	Writes data into the image after waiting on a list of Event's.
			*
			*	Writes image data from data_ptr into the image region defined by img_region. Currently channel base data type (bitness and type (uint, int or float)) and channel order of host and image
			*	must match, otherwise an exception is thrown.
			*
			*	\tparam	DepIterator		Some iterator type fulfilling the LegacyInputIterator named requirement and referring to Event objects.
			*	\param	img_region		Image region (offset and dimensions) of the target image to be written.
			*	\param	format			Host channel data type, channel order and pitch of the host memory region wher the data is read from.
			*	\param	data_ptr		Pointer to imagedata that shall be written into the specified image region.
			*	\param	dep_begin		Begin iterator of Event collection.
			*	\param	dep_end			End iterator of Event collection.
			*	\param	blocking		If true, this function blocks until the operation is finished. Otherwise, it returns immediately.
			*	\attention				Make sure data_ptr stays valid until the operation is finished when blocking is false! Otherwise this may cause access violations.
			*	\param	default_value	Currently ignored.
			*
			*	\return					Returns a Event object which can be waited upon either by other OpenCL operations or explicitely to block until the data is synchronized with OpenCL.
			*
			*	\note default_value is currently ignored until the automatic conversion feature is implemented.
			*/
			template<typename DepIterator>
			inline Event write(const ImageRegion& img_region, const HostFormat& format, const void* data_ptr, DepIterator dep_begin, DepIterator dep_end, bool blocking = true, ChannelDefaultValue default_value = ChannelDefaultValue::Zeros);
				
			/**
			*	\brief	Reads data from the image after waiting on a list of Event's.
			*
			*	Reads image data from the image region defined by img_region and stores it into the the memory region pointed by data_ptr.
			*	Currently channel base data type (bitness and type (uint, int or float)) and channel order of host and image
			*	must match, otherwise an exception is thrown.
			*
			*	\tparam	DepIterator			Some iterator type fulfilling the LegacyInputIterator named requirement and referring to Event objects.
			*	\param		img_region		Image region (offset and dimensions) of the target image to be read.
			*	\param		format			Host channel data type, channel order and pitch of the host memory region where the data should be written to.
			*	\param[out]	data_ptr		Pointer to imagedata that shall be written.
			*	\param		dep_begin		Begin iterator of Event collection.
			*	\param		dep_end			End iterator of Event collection.
			*	\param		blocking		If true, this function blocks until the operation is finished. Otherwise, it returns immediately.
			*	\attention					Make sure data_ptr stays valid until the operation is finished when blocking is false! Otherwise this may cause access violations.
			*	\param		default_value	Currently ignored.
			*
			*	\return						Returns a Event object which can be waited upon either by other OpenCL operations or explicitely to block until the data is synchronized with OpenCL.
			*
			*	\note default_value is currently ignored until the automatic conversion feature is implemented.
			*/
			template<typename DepIterator>
			inline Event read(const ImageRegion& img_region, const HostFormat& format, void* data_ptr, DepIterator dep_begin, DepIterator dep_end, bool blocking = true, ChannelDefaultValue default_value = ChannelDefaultValue::Zeros);

			/**
			*	\brief Represents a color value for e.g. filling an image with a constant color.
			*
			*	Color values are given as float 4-tupel.
			*/
			class FillColor
			{
			public:
				FillColor(float r = 0.f, float g = 0.f, float b = 0.f, float a = 0.f) : values{r, g, b, a} {}
				~FillColor() noexcept = default;
				FillColor(const FillColor&) noexcept = default;
				FillColor(FillColor&&) noexcept = default;
				FillColor& operator=(const FillColor&) noexcept = default;
				FillColor& operator=(FillColor&&) noexcept = default;

				float r() const { return values[0]; }
				float g() const { return values[1]; }
				float b() const { return values[2]; }
				float a() const { return values[3]; }

				float get(std::size_t channel_index) const
				{
					return values[channel_index];
				}

			private:
				std::array<float, 4> values;
			};

			/**
				*	\brief				Fills the specified image region with a constant color.
				*	\param color		Constant fill color.
				*	\param img_region	Region to fill within the image.
				*	\return				Returns a Event object which can be waited upon either by other OpenCL operations or explicitely to block until the data is synchronized with OpenCL.
			*/
			inline Event fill(const FillColor& color, const ImageRegion& img_region);

			/**
				*	\brief				Fills the specified image region with a constant color after waiting on a list of Event's.
				*	\tparam	DepIterator	Some iterator type fulfilling the LegacyInputIterator named requirement and referring to Event objects.
				*	\param color		Constant fill color.
				*	\param img_region	Region to fill within the image.
				*	\return				Returns a Event object which can be waited upon either by other OpenCL operations or explicitely to block until the data is synchronized with OpenCL.
			*/
			template<typename DepIterator>
			inline Event fill(const FillColor& color, const ImageRegion& img_region, DepIterator dep_begin, DepIterator dep_end);

			/**
			*	\brief Used for interfacing with Program (this class can be used as kernel argument)
			*	\return	Returns size of a cl_mem handle.
			*/
			static constexpr std::size_t arg_size() { return sizeof(cl_mem); }
			/**
			*	\brief Used for interfacing with Program (this class can be used as kernel argument)
			*	\return	Returns pointer to the cl_mem handle.
			*/
			const void* arg_data() const { return &m_image; }
		private:
			/** 
			*	\brief	Implementation of image write operations (using clEnqueueMapImage).
			*	\bug	Seems to be buggy for image2D arrays. No matter how I set origin[2], it always maps the first array slice.
			*/
			Event img_write_mapped(const ImageRegion& img_region, const HostFormat& format, const void* data_ptr, bool invalidate = false, ChannelDefaultValue default_value = ChannelDefaultValue::Zeros);
			/**
			*	\brief	Implementation of image read operations (using clEnqueueMapImage).
			*	\bug	Seems to be buggy for image2D arrays. No matter how I set origin[2], it always maps the first array slice.
			*/
			Event img_read_mapped(const ImageRegion& img_region, const HostFormat& format, void* data_ptr, ChannelDefaultValue default_value = ChannelDefaultValue::Zeros);
			/// Implementation of image write operations.
			Event img_write(const ImageRegion& img_region, const HostFormat& format, const void* data_ptr, bool blocking = true, ChannelDefaultValue default_value = ChannelDefaultValue::Zeros);
			///	Implementation of image read operations.
			Event img_read(const ImageRegion& img_region, const HostFormat& format, void* data_ptr, bool blocking = true, ChannelDefaultValue default_value = ChannelDefaultValue::Zeros);
			/// Implementation of image fill operation.
			Event img_fill(const FillColor& color, const ImageRegion& img_region);

			/// Checks whether the host format matches the image format.
			bool match_format(const HostFormat& format);
				
			cl_mem m_image;							///< Stores the OpenCL image object handle.
			ImageDesc m_image_desc;					///< Image description as passed to the constructor.
			std::vector<cl_event> m_event_cache;	///< Used to cache cl_event's in contiguous memory before calling the API functions.
			std::shared_ptr<Context> m_cl_state;	///< Shared pointer to a valid instance of Context.
		};

		inline Event simple_cl::cl::Image::write(const ImageRegion& img_region, const HostFormat& format, const void* data_ptr, bool blocking, ChannelDefaultValue default_value)
		{
			m_event_cache.clear();
			return img_write(img_region, format, data_ptr, blocking, default_value);
		}

		inline Event simple_cl::cl::Image::read(const ImageRegion& img_region, const HostFormat& format, void* data_ptr, bool blocking, ChannelDefaultValue default_value)
		{
			m_event_cache.clear();
			return img_read(img_region, format, data_ptr, blocking, default_value);
		}

		template<typename DepIterator>
		inline Event simple_cl::cl::Image::write(const ImageRegion& img_region, const HostFormat& format, const void* data_ptr, DepIterator dep_begin, DepIterator dep_end, bool blocking, ChannelDefaultValue default_value)
		{
			static_assert(std::is_same<meta::bare_type_t<typename std::iterator_traits<DepIterator>::value_type>, Event>::value, "[Image]: Dependency iterators must refer to a collection of Event objects.");
			m_event_cache.clear();
			for(DepIterator it{dep_begin}; it != dep_end; ++it)
				if(it->m_event)
					m_event_cache.push_back(it->m_event);
			return img_write_mapped(img_region, format, data_ptr, default_value);
		}

		template<typename DepIterator>
		inline Event simple_cl::cl::Image::read(const ImageRegion& img_region, const HostFormat& format, void* data_ptr, DepIterator dep_begin, DepIterator dep_end, bool blocking, ChannelDefaultValue default_value)
		{
			static_assert(std::is_same<meta::bare_type_t<typename std::iterator_traits<DepIterator>::value_type>, Event>::value, "[Image]: Dependency iterators must refer to a collection of Event objects.");
			m_event_cache.clear();
			for(DepIterator it{dep_begin}; it != dep_end; ++it)
				if(it->m_event)
					m_event_cache.push_back(it->m_event);
			return img_read_mapped(img_region, format, data_ptr, default_value);
		}

		inline std::size_t Image::get_image_channel_type_size(const Image::ImageChannelType type)
		{
			return std::size_t((static_cast<uint64_t>(type) >> 16) & uint64_t { 0x000000000000FFFF });
		}

		inline std::size_t Image::get_host_channel_type_size(const Image::HostDataType type)
		{
			return std::size_t((static_cast<uint16_t>(type) >> 4) & uint16_t { 0x000F });
		}

		inline std::size_t Image::get_num_image_pixel_components(const Image::ImageChannelOrder channel_order)
		{
			return std::size_t((static_cast<uint64_t>(channel_order) >> 24) & uint64_t { 0x00000000000000FF });
		}

		inline std::size_t Image::get_num_host_pixel_components(const Image::HostChannelOrder& channel_order)
		{
			return channel_order.num_channels;
		}

		inline cl_uint Image::get_image_channel_order_specifier(const Image::ImageChannelOrder channel_order)
		{
			return static_cast<cl_uint>((static_cast<uint64_t>(channel_order) >> 32) & uint64_t { 0x00000000FFFFFFFF });
		}

		inline cl_uint Image::get_image_channel_type_specifier(const Image::ImageChannelType channel_type)
		{
			return static_cast<cl_uint>((static_cast<uint64_t>(channel_type) >> 32) & uint64_t { 0x00000000FFFFFFFF });
		}

		inline Image::ChannelBaseType Image::get_image_channel_base_type(const Image::ImageChannelType channel_type)
		{
			return static_cast<Image::ChannelBaseType>((static_cast<uint64_t>(channel_type) >> 8) & uint64_t { 0x00000000000000FF });
		}

		inline Image::ChannelBaseType Image::get_host_channel_base_type(const Image::HostDataType channel_type)
		{
			return static_cast<Image::ChannelBaseType>(static_cast<uint16_t>(channel_type) & uint16_t { 0x000F });
		}

		inline Image::ColorChannel Image::get_image_color_channel(const Image::ImageChannelOrder channel_order, std::size_t index)
		{
			return static_cast<Image::ColorChannel>((static_cast<uint64_t>(channel_order) >> (20 - index * 4)) & 0x000000000000000F);
		}

		inline bool Image::is_image_channel_format_normalized_integer(const Image::ImageChannelType channel_type)
		{
			return static_cast<bool>(static_cast<uint64_t>(channel_type) & uint64_t { 0x00000000000000FF });
		}

		inline std::size_t Image::get_image_color_channel_index(const ImageChannelOrder channel_order, const ColorChannel channel)
		{
			std::size_t num_channels = get_num_image_pixel_components(channel_order);
			for(std::size_t i{0ull}; i < num_channels; ++i)
				if(get_image_color_channel(channel_order, i) == channel)
					return i;
			return constants::INVALID_COLOR_CHANNEL_INDEX;
		}

		inline Event Image::fill(const Image::FillColor& color, const Image::ImageRegion& img_region)
		{
			m_event_cache.clear();
			return img_fill(color, img_region);
		}

		template<typename DepIterator>
		inline Event Image::fill(const Image::FillColor& color, const Image::ImageRegion& img_region, DepIterator dep_begin, DepIterator dep_end)
		{
			m_event_cache.clear();
			for(DepIterator it{dep_begin}; it != dep_end; ++it)
				m_event_cache.push_back(it->m_event);
			return img_fill(color, img_region);
		}

		// global operators
		/// Returns true if two host channel orders match.
		inline bool operator==(const Image::HostChannelOrder& rhs, const Image::HostChannelOrder& lhs)
		{
			return ((lhs.num_channels == rhs.num_channels) &&
				(!(lhs.num_channels >= 1) || (lhs.channels[0] == rhs.channels[0])) &&
				(!(lhs.num_channels >= 2) || (lhs.channels[1] == rhs.channels[1])) &&
				(!(lhs.num_channels >= 3) || (lhs.channels[2] == rhs.channels[2])) &&
				(!(lhs.num_channels >= 4) || (lhs.channels[3] == rhs.channels[3])));
		}
		/// Returns true if two host channel orders don't match.
		inline bool operator!=(const Image::HostChannelOrder& rhs, const Image::HostChannelOrder& lhs) { return !(rhs == lhs); }
	#pragma endregion
	}
	
}

#endif