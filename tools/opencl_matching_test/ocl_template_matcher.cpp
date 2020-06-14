#include <ocl_template_matcher.hpp>
#include <CL/cl.h>
#include <ocl_error.hpp>
#include <string>


// ----------------------------------------- IMPLEMENTATION ---------------------------------------------

namespace ocl_template_matching
{
	namespace impl
	{
		namespace cl
		{
			class OpenCLState
			{
			public:
				OpenCLState(std::size_t platform_index, std::size_t device_index)
				{
					// -------------------------- setup OpenCL --------------------------------------------------
					try
					{
						cl_int cl_error{CL_SUCCESS};
						// query number of platforms available
						cl_uint number_of_platforms{0};
						CL_EX(clGetPlatformIDs(0, nullptr, &number_of_platforms));
						if(number_of_platforms == 0u)
						{
							throw std::runtime_error("[OCL_TEMPLATE_MATCHER]: This system doesn't support OpenCL.");
						}
						// query platform ID's
						std::unique_ptr<cl_platform_id[]> platform_ids(new cl_platform_id[number_of_platforms]);
						CL_EX(clGetPlatformIDs(number_of_platforms, platform_ids.get(), &number_of_platforms));
						// query platform info
						for(std::size_t p = 0; p < static_cast<std::size_t>(number_of_platforms); ++p)
						{
							// platform object for info storage
							CLPlatform platform;
							platform.id = platform_ids[p];
							// query platform info
							std::size_t infostrlen{0ull};
							std::unique_ptr<char[]> infostring;
							// profile
							CL_EX(clGetPlatformInfo(platform_ids[p], CL_PLATFORM_PROFILE, 0ull, nullptr, &infostrlen));
							infostring.reset(new char[infostrlen]);
							CL_EX(clGetPlatformInfo(platform_ids[p], CL_PLATFORM_PROFILE, infostrlen, infostring.get(), nullptr));
							platform.profile = infostring.get();
							// version
							CL_EX(clGetPlatformInfo(platform_ids[p], CL_PLATFORM_VERSION, 0ull, nullptr, &infostrlen));
							infostring.reset(new char[infostrlen]);
							CL_EX(clGetPlatformInfo(platform_ids[p], CL_PLATFORM_VERSION, infostrlen, infostring.get(), nullptr));
							platform.version = infostring.get();
							// name
							CL_EX(clGetPlatformInfo(platform_ids[p], CL_PLATFORM_NAME, 0ull, nullptr, &infostrlen));
							infostring.reset(new char[infostrlen]);
							CL_EX(clGetPlatformInfo(platform_ids[p], CL_PLATFORM_NAME, infostrlen, infostring.get(), nullptr));
							platform.name = infostring.get();
							// vendor
							CL_EX(clGetPlatformInfo(platform_ids[p], CL_PLATFORM_VENDOR, 0ull, nullptr, &infostrlen));
							infostring.reset(new char[infostrlen]);
							CL_EX(clGetPlatformInfo(platform_ids[p], CL_PLATFORM_VENDOR, infostrlen, infostring.get(), nullptr));
							platform.vendor = infostring.get();
							// extensions
							CL_EX(clGetPlatformInfo(platform_ids[p], CL_PLATFORM_EXTENSIONS, 0ull, nullptr, &infostrlen));
							infostring.reset(new char[infostrlen]);
							CL_EX(clGetPlatformInfo(platform_ids[p], CL_PLATFORM_EXTENSIONS, infostrlen, infostring.get(), nullptr));
							platform.extensions = infostring.get();

							// enumerate devices
							cl_uint num_devices;
							CL_EX(clGetDeviceIDs(platform.id, CL_DEVICE_TYPE_GPU, 0u, nullptr, &num_devices));
							// if there are no gpu devices on this platform, ignore it entirely
							if(num_devices > 0u)
							{
								std::unique_ptr<cl_device_id[]> device_ids(new cl_device_id[num_devices]);
								CL_EX(clGetDeviceIDs(platform.id, CL_DEVICE_TYPE_GPU, num_devices, device_ids.get(), nullptr));

							}
						}
					}
					catch(...)
					{
						throw;
					}
				}

			private:
				// Hold list of platforms and devices for convenience
				struct CLDevice // TODO: remove commented entries. These are mandatory for this application. Throw exception if the requirements aren't met.
				{
					cl_uint vendor_id;
					cl_uint max_compute_units;
					cl_uint max_work_item_dimensions;
					std::vector<std::size_t> max_work_item_sizes;
					std::size_t max_work_group_size;
					cl_ulong max_mem_alloc_size;
					//bool image_support;
					std::size_t image2d_max_width;
					std::size_t image2d_max_height;
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
					//bool device_available;
					//bool compiler_available;
					//bool linker_available;
					//cl_device_exec_capabilities; (CL_EXEC_KERNEL!)
					std::string name;
					std::string vendor;
					std::string driver_version;
					std::string device_profile;
					std::string device_version;
					std::string device_extensions;
					std::size_t printf_buffer_size;
				};

				struct CLPlatform
				{
					cl_platform_id id;
					std::string profile;
					std::string version;
					std::string name;
					std::string vendor;
					std::string extensions;
					std::vector<CLDevice> devices;
				};

				std::vector<CLPlatform> m_available_platforms;

				// ID's and handles for current OpenCL instance
				cl_platform_id m_current_platform;
				cl_device_id m_current_device;
				cl_context m_current_context;
				cl_command_queue m_current_command_queue;
			};


		}

		

		class MatcherImpl
		{
		public:
			MatcherImpl(const MatchingPolicyBase& matching_policy);
		private:
			cl::OpenCLState m_cl_state;
		};

		MatcherImpl::MatcherImpl(const MatchingPolicyBase& matching_policy) :
			m_cl_state{}
		{
		}
	}
}


// ----------------------------------------- INTERFACE --------------------------------------------------

ocl_template_matching::impl::MatcherBase::MatcherBase(const MatchingPolicyBase& matching_policy) :
	m_impl(std::make_unique<MatcherImpl>(matching_policy))
{

}

ocl_template_matching::impl::MatcherBase::MatcherBase(MatcherBase&& other) noexcept = default;

ocl_template_matching::impl::MatcherBase& ocl_template_matching::impl::MatcherBase::operator=(MatcherBase&& other) noexcept = default;

ocl_template_matching::impl::MatcherBase::~MatcherBase()
{
}

ocl_template_matching::MatchingResult ocl_template_matching::impl::MatcherBase::match(const Texture& texture,
	const cv::Mat& texture_mask,
	const Texture& kernel,
	const cv::Mat& kernel_mask,
	const MatchingPolicyBase& matching_policy)
{
	return ocl_template_matching::MatchingResult{};
}

void ocl_template_matching::impl::MatcherBase::match(const Texture& texture,
	const cv::Mat& texture_mask,
	const Texture& kernel,
	const cv::Mat& kernel_mask,
	MatchingResult& result,
	const MatchingPolicyBase& matching_policy)
{

}