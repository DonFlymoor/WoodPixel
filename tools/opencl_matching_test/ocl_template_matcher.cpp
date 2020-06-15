#include <ocl_template_matcher.hpp>
#include <CL/cl.h>
#include <ocl_error.hpp>
#include <string>
#include <iostream>


// ----------------------------------------- IMPLEMENTATION ---------------------------------------------

namespace ocl_template_matching
{
	namespace impl
	{
		namespace util
		{
			std::vector<std::string> string_split(const std::string& s, char delimiter)
			{
				std::vector<std::string> tokens;
				std::string token;
				std::istringstream sstr(s);
				while(std::getline(sstr, token, delimiter))
				{
					tokens.push_back(token);
				}
				return tokens;
			}
		}

		namespace cl
		{
			class CLState
			{
			public:
				CLState(std::size_t platform_index, std::size_t device_index)
				{
					try
					{
						read_platform_and_device_info();
						print_suitable_platform_and_device_info();
						init_cl_instance(platform_index, device_index);
					}
					catch(...)
					{
						std::cerr << "[ERROR][OCL_TEMPLATE_MATCHER]: OpenCL initialization failed." << std::endl;
						throw;
					}
				}

			private:

				// --- private types

				// Hold list of platforms and devices for convenience
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

				// --- private data members

				std::vector<CLPlatform> m_available_platforms;

				// ID's and handles for current OpenCL instance
				cl_platform_id m_current_platform;
				cl_device_id m_current_device;
				cl_context m_current_context;
				cl_command_queue m_current_command_queue;

				// --- private member functions

				// friends
				friend std::ostream& operator<<(std::ostream&, const CLState::CLPlatform&);
				friend std::ostream& operator<<(std::ostream&, const CLState::CLDevice&);

				void print_suitable_platform_and_device_info()
				{
					std::cout << "===== SUITABLE OpenCL PLATFORMS AND DEVICES =====" << std::endl;
					for(std::size_t p = 0ull; p < m_available_platforms.size(); ++p)
					{
						std::cout << "[Platform ID: " << p << "] " << m_available_platforms[p] << std::endl;
					}
				}

				void read_platform_and_device_info()
				{					
					// query number of platforms available
					cl_uint number_of_platforms{0};
					CL_EX(clGetPlatformIDs(0, nullptr, &number_of_platforms));
					if(number_of_platforms == 0u)
						return;
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
						unsigned int plat_version_identifier{get_cl_version_num(platform.version)};
						if(plat_version_identifier < 120)
							continue;
						platform.version_num = plat_version_identifier;
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

							// query device info and store suitable ones 
							for(size_t d = 0; d < num_devices; ++d)
							{
								CLDevice device;
								// device id
								device.device_id = device_ids[d];
								// --- check if device is suitable
								// device version
								CL_EX(clGetDeviceInfo(device_ids[d], CL_DEVICE_VERSION, 0ull, nullptr, &infostrlen));
								infostring.reset(new char[infostrlen]);
								CL_EX(clGetDeviceInfo(device_ids[d], CL_DEVICE_VERSION, infostrlen, infostring.get(), nullptr));
								device.device_version = infostring.get();
								// check if device version >= 1.2
								unsigned int version_identifier{get_cl_version_num(device.device_version)};
								if(version_identifier < 120u)
									continue;
								device.device_version_num = version_identifier;
								// image support
								cl_bool image_support;
								CL_EX(clGetDeviceInfo(device_ids[d], CL_DEVICE_IMAGE_SUPPORT, sizeof(cl_bool), &image_support, nullptr));
								if(!image_support)
									continue;
								// device available
								cl_bool device_available;
								CL_EX(clGetDeviceInfo(device_ids[d], CL_DEVICE_AVAILABLE, sizeof(cl_bool), &device_available, nullptr));
								if(!device_available)
									continue;
								// compiler available
								cl_bool compiler_available;
								CL_EX(clGetDeviceInfo(device_ids[d], CL_DEVICE_COMPILER_AVAILABLE, sizeof(cl_bool), &compiler_available, nullptr));
								if(!compiler_available)
									continue;
								// linker available
								cl_bool linker_available;
								CL_EX(clGetDeviceInfo(device_ids[d], CL_DEVICE_LINKER_AVAILABLE, sizeof(cl_bool), &linker_available, nullptr));
								if(!linker_available)
									continue;
								// exec capabilities
								cl_device_exec_capabilities exec_capabilities;
								CL_EX(clGetDeviceInfo(device_ids[d], CL_DEVICE_EXECUTION_CAPABILITIES, sizeof(cl_device_exec_capabilities), &exec_capabilities, nullptr));
								if(!(exec_capabilities | CL_EXEC_KERNEL))
									continue;

								// --- additional info
								// vendor id
								CL_EX(clGetDeviceInfo(device_ids[d], CL_DEVICE_VENDOR_ID, sizeof(cl_uint), &device.vendor_id, nullptr));
								// max compute units
								CL_EX(clGetDeviceInfo(device_ids[d], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &device.max_compute_units, nullptr));
								// max work item dimensions
								CL_EX(clGetDeviceInfo(device_ids[d], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &device.max_work_item_dimensions, nullptr));
								// max work item sizes
								device.max_work_item_sizes = std::vector<std::size_t>(static_cast<std::size_t>(device.max_work_item_dimensions), 0);
								CL_EX(clGetDeviceInfo(device_ids[d], CL_DEVICE_MAX_WORK_ITEM_SIZES, device.max_work_item_sizes.size() * sizeof(std::size_t), device.max_work_item_sizes.data(), nullptr));
								// max work group size
								CL_EX(clGetDeviceInfo(device_ids[d], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(std::size_t), &device.max_work_group_size, nullptr));
								// max mem alloc size
								CL_EX(clGetDeviceInfo(device_ids[d], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &device.max_mem_alloc_size, nullptr));
								// image2d max width
								CL_EX(clGetDeviceInfo(device_ids[d], CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(std::size_t), &device.image2d_max_width, nullptr));
								// image2d max height
								CL_EX(clGetDeviceInfo(device_ids[d], CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(std::size_t), &device.image2d_max_height, nullptr));
								// image3d max width
								CL_EX(clGetDeviceInfo(device_ids[d], CL_DEVICE_IMAGE3D_MAX_WIDTH, sizeof(std::size_t), &device.image3d_max_width, nullptr));
								// image3d max height
								CL_EX(clGetDeviceInfo(device_ids[d], CL_DEVICE_IMAGE3D_MAX_HEIGHT, sizeof(std::size_t), &device.image3d_max_height, nullptr));
								// image3d max depth
								CL_EX(clGetDeviceInfo(device_ids[d], CL_DEVICE_IMAGE3D_MAX_DEPTH, sizeof(std::size_t), &device.image3d_max_depth, nullptr));
								// image max buffer size
								CL_EX(clGetDeviceInfo(device_ids[d], CL_DEVICE_IMAGE_MAX_BUFFER_SIZE, sizeof(std::size_t), &device.image_max_buffer_size, nullptr));
								// image max array size
								CL_EX(clGetDeviceInfo(device_ids[d], CL_DEVICE_IMAGE_MAX_ARRAY_SIZE, sizeof(std::size_t), &device.image_max_array_size, nullptr));
								// max samplers
								CL_EX(clGetDeviceInfo(device_ids[d], CL_DEVICE_MAX_SAMPLERS, sizeof(cl_uint), &device.max_samplers, nullptr));
								// max parameter size
								CL_EX(clGetDeviceInfo(device_ids[d], CL_DEVICE_MAX_PARAMETER_SIZE, sizeof(std::size_t), &device.max_parameter_size, nullptr));
								// mem base addr align
								CL_EX(clGetDeviceInfo(device_ids[d], CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(cl_uint), &device.mem_base_addr_align, nullptr));
								// global mem cacheline size
								CL_EX(clGetDeviceInfo(device_ids[d], CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(cl_uint), &device.global_mem_cacheline_size, nullptr));
								// global mem cache size
								CL_EX(clGetDeviceInfo(device_ids[d], CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(cl_ulong), &device.global_mem_cache_size, nullptr));
								// global mem size
								CL_EX(clGetDeviceInfo(device_ids[d], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &device.global_mem_size, nullptr));
								// max constant buffer size
								CL_EX(clGetDeviceInfo(device_ids[d], CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(cl_ulong), &device.max_constant_buffer_size, nullptr));
								// max constant args
								CL_EX(clGetDeviceInfo(device_ids[d], CL_DEVICE_MAX_CONSTANT_ARGS, sizeof(cl_uint), &device.max_constant_args, nullptr));
								// local mem size
								CL_EX(clGetDeviceInfo(device_ids[d], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &device.local_mem_size, nullptr));
								// little or big endian
								cl_bool little_end;
								CL_EX(clGetDeviceInfo(device_ids[d], CL_DEVICE_ENDIAN_LITTLE, sizeof(cl_bool), &little_end, nullptr));
								device.little_endian = (little_end == CL_TRUE);
								// name
								CL_EX(clGetDeviceInfo(device_ids[d], CL_DEVICE_NAME, 0ull, nullptr, &infostrlen));
								infostring.reset(new char[infostrlen]);
								CL_EX(clGetDeviceInfo(device_ids[d], CL_DEVICE_NAME, infostrlen, infostring.get(), nullptr));
								device.name = infostring.get();
								// vendor
								CL_EX(clGetDeviceInfo(device_ids[d], CL_DEVICE_VENDOR, 0ull, nullptr, &infostrlen));
								infostring.reset(new char[infostrlen]);
								CL_EX(clGetDeviceInfo(device_ids[d], CL_DEVICE_VENDOR, infostrlen, infostring.get(), nullptr));
								device.vendor = infostring.get();
								// driver version
								CL_EX(clGetDeviceInfo(device_ids[d], CL_DRIVER_VERSION, 0ull, nullptr, &infostrlen));
								infostring.reset(new char[infostrlen]);
								CL_EX(clGetDeviceInfo(device_ids[d], CL_DRIVER_VERSION, infostrlen, infostring.get(), nullptr));
								device.driver_version = infostring.get();
								// device profile
								CL_EX(clGetDeviceInfo(device_ids[d], CL_DEVICE_PROFILE, 0ull, nullptr, &infostrlen));
								infostring.reset(new char[infostrlen]);
								CL_EX(clGetDeviceInfo(device_ids[d], CL_DEVICE_PROFILE, infostrlen, infostring.get(), nullptr));
								device.device_profile = infostring.get();
								// device extensions
								CL_EX(clGetDeviceInfo(device_ids[d], CL_DEVICE_EXTENSIONS, 0ull, nullptr, &infostrlen));
								infostring.reset(new char[infostrlen]);
								CL_EX(clGetDeviceInfo(device_ids[d], CL_DEVICE_EXTENSIONS, infostrlen, infostring.get(), nullptr));
								device.device_extensions = infostring.get();
								// printf buffer size
								CL_EX(clGetDeviceInfo(device_ids[d], CL_DEVICE_PRINTF_BUFFER_SIZE, sizeof(std::size_t), &device.printf_buffer_size, nullptr));

								// success! Add the device to the list of suitable devices of the platform.
								platform.devices.push_back(std::move(device));
							}
							// if there are suitable devices, add this platform to the list of suitable platforms.
							if(platform.devices.size() > 0)
							{
								m_available_platforms.push_back(std::move(platform));
							}
						}
					}					
				}

				unsigned int get_cl_version_num(const std::string& str)
				{
					std::string version_string = util::string_split(str, ' ')[1];
					unsigned int version_major = static_cast<unsigned int>(std::stoul(util::string_split(version_string, '.')[0]));
					unsigned int version_minor = static_cast<unsigned int>(std::stoul(util::string_split(version_string, '.')[1]));
					return version_major * 100u + version_minor * 10u;
				}
			
				void init_cl_instance(std::size_t platform_id, std::size_t device_id)
				{
					if(m_available_platforms.size() == 0ull)
						throw std::runtime_error("[OCL_TEMPLATE_MATCHER]: No suitable OpenCL 1.2 platform found.");
					if(platform_id > m_available_platforms.size())
						throw std::runtime_error("[OCL_TEMPLATE_MATCHER]: Platform index out of range.");
					if(m_available_platforms[platform_id].devices.size() == 0ull)
						throw std::runtime_error("[OCL_TEMPLATE_MATCHER]: No suitable OpenCL 1.2 device found.");
					if(device_id > m_available_platforms[platform_id].devices.size())
						throw std::runtime_error("[OCL_TEMPLATE_MATCHER]: Device index out of range.");
				}
			};

			std::ostream& operator<<(std::ostream& os, const CLState::CLPlatform& plat)
			{
				os	<< "===== OpenCL Platform =====" << std::endl
					<< "Name:" << std::endl
					<< "\t" << plat.name << std::endl
					<< "Vendor:" << std::endl
					<< "\t" << plat.vendor << std::endl
					<< "Version:" << std::endl
					<< "\t" << plat.version << std::endl
					<< "Profile:" << std::endl
					<< "\t" << plat.profile << std::endl
					<< "Extensions:" << std::endl
					<< "\t" << plat.extensions << std::endl
					<< std::endl
					<< "Suitable OpenCL 1.2+ devices:" << std::endl;
				for(std::size_t d = 0ull; d < plat.devices.size(); ++d)
				{
					os << std::endl;
					os << "[Device ID: "<< d << "] " << plat.devices[d];
				}
				return os;
			}

			std::ostream& operator<<(std::ostream& os, const CLState::CLDevice& dev)
			{
				os << "===== OpenCL Device =====" << std::endl
					<< "Vendor ID:" << std::endl
					<< "\t" << dev.vendor_id << std::endl
					<< "Name:" << std::endl
					<< "\t" << dev.name << std::endl
					<< "Vendor:" << std::endl
					<< "\t" << dev.vendor << std::endl
					<< "Driver version:" << std::endl
					<< "\t" << dev.driver_version << std::endl
					<< "Device profile:" << std::endl
					<< "\t" << dev.device_profile << std::endl
					<< "Device version:" << std::endl
					<< "\t" << dev.device_version << std::endl
					<< "Max. compute units:" << std::endl
					<< "\t" << dev.max_compute_units << std::endl
					<< "Max. work item dimensions:" << std::endl
					<< "\t" << dev.max_work_item_dimensions << std::endl
					<< "Max. work item sizes:" << std::endl
					<< "\t{ ";
				for(const std::size_t& s : dev.max_work_item_sizes)
					os << s << " ";
				os	<< "}" << std::endl
					<< "Max. work group size:" << std::endl
					<< "\t" << dev.max_work_group_size << std::endl
					<< "Max. memory allocation size:" << std::endl
					<< "\t" << dev.max_mem_alloc_size << " bytes" << std::endl
					<< "Image2D max. width:" << std::endl
					<< "\t" << dev.image2d_max_width << std::endl
					<< "Image2D max. height:" << std::endl
					<< "\t" << dev.image2d_max_height << std::endl
					<< "Image3D max. width:" << std::endl
					<< "\t" << dev.image3d_max_width << std::endl
					<< "Image3D max. height:" << std::endl
					<< "\t" << dev.image3d_max_height << std::endl
					<< "Image3D max. depth:" << std::endl
					<< "\t" << dev.image3d_max_depth << std::endl
					<< "Image max. buffer size:" << std::endl
					<< "\t" << dev.image_max_buffer_size << std::endl
					<< "Image max. array size:" << std::endl
					<< "\t" << dev.image_max_array_size << std::endl
					<< "Max. samplers:" << std::endl
					<< "\t" << dev.max_samplers << std::endl
					<< "Max. parameter size:" << std::endl
					<< "\t" << dev.max_parameter_size << " bytes" << std::endl
					<< "Memory base address alignment:" << std::endl
					<< "\t" << dev.mem_base_addr_align << " bytes" << std::endl
					<< "Global memory cache line size:" << std::endl
					<< "\t" << dev.global_mem_cacheline_size << " bytes" << std::endl
					<< "Global memory cache size:" << std::endl
					<< "\t" << dev.global_mem_cache_size << " bytes" << std::endl
					<< "Global memory size:" << std::endl
					<< "\t" << dev.global_mem_size << " bytes" << std::endl
					<< "Max. constant buffer size:" << std::endl
					<< "\t" << dev.max_constant_buffer_size << " bytes" << std::endl
					<< "Max. constant args:" << std::endl
					<< "\t" << dev.max_constant_args << std::endl
					<< "Local memory size:" << std::endl
					<< "\t" << dev.local_mem_size << " bytes" << std::endl
					<< "Little endian:" << std::endl
					<< "\t" << (dev.little_endian ? "yes" : "no") << std::endl
					<< "printf buffer size:" << std::endl
					<< "\t" << dev.printf_buffer_size << " bytes" << std::endl
					<< "Extensions:" << std::endl
					<< "\t" << dev.device_extensions << std::endl;
				return os;
			}
		}		

		class MatcherImpl
		{
		public:
			MatcherImpl(const MatchingPolicyBase& matching_policy);
		private:
			cl::CLState m_cl_state;
		};

		MatcherImpl::MatcherImpl(const MatchingPolicyBase& matching_policy) :
			m_cl_state{matching_policy.platform_id(), matching_policy.device_id()}
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