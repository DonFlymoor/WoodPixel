#include <ocl_wrappers.hpp>

// -------------------------------------------- NAMESPACE ocl_template_matching::impl::util-----------------------------------
#pragma region util

std::vector<std::string> ocl_template_matching::impl::util::string_split(const std::string& s, char delimiter)
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

unsigned int ocl_template_matching::impl::util::get_cl_version_num(const std::string& str)
{
	std::string version_string = util::string_split(str, ' ')[1];
	unsigned int version_major = static_cast<unsigned int>(std::stoul(util::string_split(version_string, '.')[0]));
	unsigned int version_minor = static_cast<unsigned int>(std::stoul(util::string_split(version_string, '.')[1]));
	return version_major * 100u + version_minor * 10u;
}

#pragma endregion

// -------------------------------------------- NAMESPACE ocl_template_matching::impl::cl -------------------------------------

#pragma region cl
#pragma region class CLState
// ---------------------- class CLState
// factory function
std::shared_ptr<ocl_template_matching::impl::cl::CLState> ocl_template_matching::impl::cl::createCLInstance(std::size_t platform_index, std::size_t device_index)
{
	return std::shared_ptr<CLState>(new CLState{platform_index, device_index});
}

ocl_template_matching::impl::cl::CLState::CLState(std::size_t platform_index, std::size_t device_index) :
	m_available_platforms{},
	m_selected_platform_index{0},
	m_selected_device_index{0},
	m_context{nullptr},
	m_command_queue{nullptr},
	m_cl_ex_holder{nullptr}		
{
	try
	{
		read_platform_and_device_info();
		print_suitable_platform_and_device_info();
		init_cl_instance(platform_index, device_index);
	}
	catch(...)
	{
		cleanup();
		std::cerr << "[ERROR][OCL_TEMPLATE_MATCHER]: OpenCL initialization failed." << std::endl;
		throw;
	}
}

ocl_template_matching::impl::cl::CLState::~CLState()
{
	cleanup();
}

ocl_template_matching::impl::cl::CLState::CLState(CLState&& other) noexcept :
	m_available_platforms(std::move(other.m_available_platforms)),
	m_selected_platform_index{other.m_selected_platform_index},
	m_selected_device_index{other.m_selected_device_index},
	m_context{other.m_context},
	m_command_queue{other.m_command_queue},
	m_cl_ex_holder{std::move(other.m_cl_ex_holder)}
{
	other.m_command_queue = nullptr;
	other.m_context = nullptr;
	other.m_cl_ex_holder.ex_msg = nullptr;
}

ocl_template_matching::impl::cl::CLState& ocl_template_matching::impl::cl::CLState::operator=(CLState&& other) noexcept
{
	if(this == &other)
		return *this;

	cleanup();

	m_available_platforms = std::move(other.m_available_platforms);
	m_selected_platform_index = other.m_selected_platform_index;
	m_selected_device_index = other.m_selected_device_index;
	std::swap(m_context, other.m_context);
	std::swap(m_command_queue, other.m_command_queue);
	std::swap(m_cl_ex_holder, other.m_cl_ex_holder);

	return *this;
}

void ocl_template_matching::impl::cl::CLState::print_selected_platform_info() const
{
	std::cout << "===== Selected OpenCL platform =====" << std::endl;
	std::cout << m_available_platforms[m_selected_platform_index];
}

void ocl_template_matching::impl::cl::CLState::print_selected_device_info() const
{
	std::cout << "===== Selected OpenCL device =====" << std::endl;
	std::cout << m_available_platforms[m_selected_platform_index].devices[m_selected_device_index];
}

void ocl_template_matching::impl::cl::CLState::print_suitable_platform_and_device_info() const
{
	std::cout << "===== SUITABLE OpenCL PLATFORMS AND DEVICES =====" << std::endl;
	for(std::size_t p = 0ull; p < m_available_platforms.size(); ++p)
	{
		std::cout << "[Platform ID: " << p << "] " << m_available_platforms[p] << std::endl;
		std::cout << "Suitable OpenCL 1.2+ devices:" << std::endl;
		for(std::size_t d = 0ull; d < m_available_platforms[p].devices.size(); ++d)
		{
			std::cout << std::endl;
			std::cout << "[Platform ID: " << p << "]" << "[Device ID: " << d << "] " << m_available_platforms[p].devices[d];
		}
	}
}

void ocl_template_matching::impl::cl::CLState::read_platform_and_device_info()
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
		unsigned int plat_version_identifier{util::get_cl_version_num(platform.version)};
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
				unsigned int version_identifier{util::get_cl_version_num(device.device_version)};
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

void ocl_template_matching::impl::cl::CLState::init_cl_instance(std::size_t platform_id, std::size_t device_id)
{
	if(m_available_platforms.size() == 0ull)
		throw std::runtime_error("[OCL_TEMPLATE_MATCHER]: No suitable OpenCL 1.2 platform found.");
	if(platform_id > m_available_platforms.size())
		throw std::runtime_error("[OCL_TEMPLATE_MATCHER]: Platform index out of range.");
	if(m_available_platforms[platform_id].devices.size() == 0ull)
		throw std::runtime_error("[OCL_TEMPLATE_MATCHER]: No suitable OpenCL 1.2 device found.");
	if(device_id > m_available_platforms[platform_id].devices.size())
		throw std::runtime_error("[OCL_TEMPLATE_MATCHER]: Device index out of range.");

	// select device and platform
	// TODO: Future me: maybe use all available devices of one platform? Would be nice to have the option...
	m_selected_platform_index = platform_id;
	m_selected_device_index = device_id;

	std::cout << std::endl << "========== OPENCL INITIALIZATION ==========" << std::endl;
	std::cout << "Selected platform ID: " << m_selected_platform_index << std::endl;
	std::cout << "Selected device ID: " << m_selected_device_index << std::endl << std::endl;

	// create OpenCL context
	std::cout << "Creating OpenCL context...";
	cl_context_properties ctprops[]{
		CL_CONTEXT_PLATFORM,
		reinterpret_cast<cl_context_properties>(m_available_platforms[m_selected_platform_index].id),
		0
	};
	cl_int res;
	m_context = clCreateContext(&ctprops[0],
		1u,
		&m_available_platforms[m_selected_platform_index].devices[m_selected_device_index].device_id,
		&create_context_callback,
		&m_cl_ex_holder,
		&res
	);
	// if an error occured during context creation, throw an appropriate exception.
	if(res != CL_SUCCESS)
		throw CLException(res, __LINE__, __FILE__, m_cl_ex_holder.ex_msg);
	std::cout << " done!" << std::endl;

	// create command queue
	std::cout << "Creating command queue...";
	m_command_queue = clCreateCommandQueue(m_context,
		m_available_platforms[m_selected_platform_index].devices[m_selected_device_index].device_id,
		cl_command_queue_properties{0ull},
		&res
	);
	if(res != CL_SUCCESS)
		throw CLException(res, __LINE__, __FILE__, "Command queue creation failed.");
	std::cout << " done!" << std::endl;
}

void ocl_template_matching::impl::cl::CLState::cleanup()
{
	if(m_command_queue)
		CL(clReleaseCommandQueue(m_command_queue));
	m_command_queue = nullptr;
	if(m_context)
		CL(clReleaseContext(m_context));
	m_context = nullptr;
	m_cl_ex_holder.ex_msg = nullptr;
}

const ocl_template_matching::impl::cl::CLState::CLPlatform& ocl_template_matching::impl::cl::CLState::get_selected_platform() const
{
	return m_available_platforms[m_selected_platform_index];
}

const ocl_template_matching::impl::cl::CLState::CLDevice& ocl_template_matching::impl::cl::CLState::get_selected_device() const
{
	return m_available_platforms[m_selected_platform_index].devices[m_selected_device_index];
}

std::ostream& ocl_template_matching::impl::cl::operator<<(std::ostream& os, const ocl_template_matching::impl::cl::CLState::CLPlatform& plat)
{
	os << "===== OpenCL Platform =====" << std::endl
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
		<< std::endl;
	return os;
}

std::ostream& ocl_template_matching::impl::cl::operator<<(std::ostream& os, const ocl_template_matching::impl::cl::CLState::CLDevice& dev)
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
	os << "}" << std::endl
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

// opencl callbacks

void ocl_template_matching::impl::cl::create_context_callback(const char* errinfo, const void* private_info, std::size_t cb, void* user_data)
{
	static_cast<CLState::CLExHolder*>(user_data)->ex_msg = errinfo;
}

#pragma endregion

#pragma region class CLProgram
// -------------------------- class CLProgram

ocl_template_matching::impl::cl::CLProgram::CLProgram(const std::string& kernel_source, const std::string& compiler_options, const std::shared_ptr<CLState>& clstate) :
	m_source(kernel_source),
	m_kernels(),
	m_cl_state(clstate),
	m_cl_program(nullptr),
	m_options(compiler_options),
	m_event_cache()
{
	try
	{
		// create program
		const char* source = m_source.data();
		std::size_t source_len = m_source.size();
		cl_int res;
		m_cl_program = clCreateProgramWithSource(m_cl_state->context(), 1u, &source, &source_len, &res);
		if(res != CL_SUCCESS)
			throw CLException{res, __LINE__, __FILE__, "clCreateProgramWithSource failed."};
		
		// build program // TODO: Multiple devices?
		res = clBuildProgram(m_cl_program, 1u, &m_cl_state->get_selected_device().device_id, m_options.data(), nullptr, nullptr);
		if(res != CL_SUCCESS)
		{
			if(res == CL_BUILD_PROGRAM_FAILURE)
			{
				std::size_t log_size{0};
				CL_EX(clGetProgramBuildInfo(m_cl_program, m_cl_state->get_selected_device().device_id, CL_PROGRAM_BUILD_LOG, 0ull, nullptr, &log_size));
				std::unique_ptr<char[]> infostring{new char[log_size]};
				CL_EX(clGetProgramBuildInfo(m_cl_program, m_cl_state->get_selected_device().device_id, CL_PROGRAM_BUILD_LOG, log_size, infostring.get(), nullptr));
				std::cerr << "OpenCL program build failed:" << std::endl << infostring.get() << std::endl;
				throw CLException{res, __LINE__, __FILE__, "OpenCL program build failed."};
			}
			else
			{
				throw CLException{res, __LINE__, __FILE__, "clBuildProgram failed."};
			}
		}

		// extract kernels and parameters
		std::size_t num_kernels{0};
		CL_EX(clGetProgramInfo(m_cl_program, CL_PROGRAM_NUM_KERNELS, sizeof(std::size_t), &num_kernels, nullptr));
		std::size_t kernel_name_string_length{0};
		CL_EX(clGetProgramInfo(m_cl_program, CL_PROGRAM_KERNEL_NAMES, 0ull, nullptr, &kernel_name_string_length));
		std::unique_ptr<char[]> kernel_name_string{new char[kernel_name_string_length]};
		CL_EX(clGetProgramInfo(m_cl_program, CL_PROGRAM_KERNEL_NAMES, kernel_name_string_length, kernel_name_string.get(), nullptr));
		std::vector<std::string> kernel_names{util::string_split(std::string{kernel_name_string.get()}, ';')};
		if(kernel_names.size() != num_kernels)
			throw std::logic_error("Number of kernels in program does not match reported number of kernels.");

		// create kernels
		for(std::size_t i = 0; i < num_kernels; ++i)
		{
			cl_kernel kernel = clCreateKernel(m_cl_program, kernel_names[i].c_str(), &res); if(res != CL_SUCCESS) throw CLException{res, __LINE__, __FILE__, "clCreateKernel failed."};
			m_kernels[kernel_names[i]] = CLKernel{i, kernel};
		}
	}
	catch(...)
	{
		cleanup();
		throw;
	}
}

ocl_template_matching::impl::cl::CLProgram::~CLProgram()
{
	cleanup();
}

ocl_template_matching::impl::cl::CLProgram::CLProgram(CLProgram&& other) noexcept :
	m_source{std::move(other.m_source)},
	m_kernels{std::move(other.m_kernels)},
	m_cl_state{std::move(other.m_cl_state)},
	m_cl_program{other.m_cl_program},
	m_options{std::move(other.m_options)},
	m_event_cache{std::move(other.m_event_cache)}
{
	m_event_cache.clear();
	other.m_kernels.clear();
	other.m_cl_program = nullptr;	
}

ocl_template_matching::impl::cl::CLProgram& ocl_template_matching::impl::cl::CLProgram::operator=(CLProgram&& other) noexcept
{
	if(this == &other)
		return *this;

	cleanup();
	m_cl_state = std::move(other.m_cl_state);
	m_source = std::move(other.m_source);
	m_options = std::move(other.m_options);
	std::swap(m_kernels, other.m_kernels);
	std::swap(m_cl_program, other.m_cl_program);
	m_event_cache.clear();
	other.m_event_cache.clear();
	std::swap(m_event_cache, other.m_event_cache);

	return *this;
}

void ocl_template_matching::impl::cl::CLProgram::cleanup() noexcept
{
	for(auto& k : m_kernels)
	{
		if(k.second.kernel)
			clReleaseKernel(k.second.kernel);
	}
	m_kernels.clear();
	if(m_cl_program)
		clReleaseProgram(m_cl_program);
}

ocl_template_matching::impl::cl::CLEvent ocl_template_matching::impl::cl::CLProgram::invoke(cl_kernel kernel, const std::vector<cl_event>& dep_events, const ExecParams& exparams)
{
	cl_event ev{nullptr};
	CL_EX(clEnqueueNDRangeKernel(
		m_cl_state->command_queue(),
		kernel,
		static_cast<cl_uint>(exparams.work_dim),
		exparams.work_offset,
		exparams.global_work_size,
		exparams.local_work_size,
		static_cast<cl_uint>(dep_events.size()),
		dep_events.size() > 0ull ? dep_events.data() : nullptr,
		&ev
	));
	return CLEvent{ev};
}

void ocl_template_matching::impl::cl::CLProgram::setKernelArgsImpl(const std::string& name, std::size_t index, std::size_t arg_size, const void* arg_data_ptr)
{
	try
	{
		CL_EX(clSetKernelArg(m_kernels.at(name).kernel, static_cast<cl_uint>(index), arg_size, arg_data_ptr));
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

void ocl_template_matching::impl::cl::CLProgram::setKernelArgsImpl(cl_kernel kernel, std::size_t index, std::size_t arg_size, const void* arg_data_ptr)
{	
	CL_EX(clSetKernelArg(kernel, static_cast<cl_uint>(index), arg_size, arg_data_ptr));
}

ocl_template_matching::impl::cl::CLProgram::CLKernelHandle ocl_template_matching::impl::cl::CLProgram::getKernel(const std::string& name)
{
	try
	{
		return CLKernelHandle{m_kernels.at(name).kernel};
	}
	catch(const std::out_of_range&)
	{
		throw std::runtime_error("Unknown kernel name.");
	}
}

#pragma endregion

#pragma region class CLEvent
// class CLEvent
ocl_template_matching::impl::cl::CLEvent::CLEvent(cl_event ev) :
	m_event{ev}
{	
}

ocl_template_matching::impl::cl::CLEvent::~CLEvent()
{
	if(m_event)
		CL_EX(clReleaseEvent(m_event));
}

ocl_template_matching::impl::cl::CLEvent::CLEvent(const CLEvent& other) :
	m_event{other.m_event}
{
	if(m_event)
		CL_EX(clRetainEvent(m_event));
}

ocl_template_matching::impl::cl::CLEvent::CLEvent(CLEvent&& other) noexcept :
	m_event{other.m_event}
{
	other.m_event = nullptr;
}

ocl_template_matching::impl::cl::CLEvent& ocl_template_matching::impl::cl::CLEvent::operator=(const CLEvent& other)
{
	if(this == &other)
		return *this;

	m_event = other.m_event;
	CL_EX(clRetainEvent(m_event));

	return *this;
}

ocl_template_matching::impl::cl::CLEvent& ocl_template_matching::impl::cl::CLEvent::operator=(CLEvent&& other) noexcept
{
	if(this == &other)
		return *this;

	std::swap(m_event, other.m_event);

	return *this;
}

void ocl_template_matching::impl::cl::CLEvent::wait() const
{
	CL_EX(clWaitForEvents(1, &m_event));
}

#pragma endregion

#pragma region class CLBuffer
// class CLBuffer

ocl_template_matching::impl::cl::CLBuffer::CLBuffer(std::size_t size, cl_mem_flags flags, const std::shared_ptr<CLState>& clstate, void* hostptr) :
	m_cl_memory{nullptr},
	m_size{0ull},
	m_cl_state{clstate},
	m_flags{0},
	m_hostptr{nullptr},
	m_event_cache{}
{	
	cl_int err{CL_SUCCESS};
	m_cl_memory = clCreateBuffer(m_cl_state->context(), flags, size, hostptr, &err);
	if(err != CL_SUCCESS)
		throw CLException(err, __LINE__, __FILE__, "[CLBuffer]: OpenCL buffer creation failed.");
	m_size = size;
	m_flags = flags;
	m_hostptr = hostptr;
}

ocl_template_matching::impl::cl::CLBuffer::~CLBuffer() noexcept
{
	if(m_cl_memory)
		CL(clReleaseMemObject(m_cl_memory));
}

ocl_template_matching::impl::cl::CLBuffer::CLBuffer(CLBuffer&& other) noexcept :
	m_cl_memory{nullptr},
	m_size{0ull},
	m_cl_state{nullptr},
	m_flags{0},
	m_hostptr{nullptr},
	m_event_cache{}
{
	std::swap(m_cl_memory, other.m_cl_memory);
	std::swap(m_size, other.m_size);
	std::swap(m_cl_state, other.m_cl_state);
	std::swap(m_flags, other.m_flags);
	std::swap(m_hostptr, other.m_hostptr);
}

ocl_template_matching::impl::cl::CLBuffer& ocl_template_matching::impl::cl::CLBuffer::operator=(CLBuffer&& other) noexcept
{
	if(this == &other)
		return *this;
	
	std::swap(m_cl_memory, other.m_cl_memory);
	std::swap(m_size, other.m_size);
	std::swap(m_cl_state, other.m_cl_state);
	std::swap(m_flags, other.m_flags);
	std::swap(m_hostptr, other.m_hostptr);
	m_event_cache.clear();

	return *this;
}

ocl_template_matching::impl::cl::CLEvent ocl_template_matching::impl::cl::CLBuffer::buf_write(const void* data, std::size_t length, std::size_t offset, bool invalidate)
{
	if(offset + length > m_size)
		throw std::out_of_range("[CLBuffer]: Buffer write failed. Input offset + length out of range.");
	if(m_flags | CL_MEM_HOST_READ_ONLY | CL_MEM_HOST_NO_ACCESS)
		throw std::runtime_error("[CLBuffer]: Writing to a read only buffer is not allowed.");
	std::size_t _offset = (length > 0ull ? offset : 0ull);
	std::size_t _length = (length > 0ull ? length : m_size);
	cl_int err{CL_SUCCESS};
	cl_event unmap_event{nullptr};
	void* bufptr = clEnqueueMapBuffer(m_cl_state->command_queue(), m_cl_memory, true, (invalidate ? CL_MAP_WRITE_INVALIDATE_REGION : CL_MAP_WRITE), _offset, _length, static_cast<cl_uint>(m_event_cache.size()), (m_event_cache.size() > 0ull? m_event_cache.data() : nullptr), nullptr, &err);
	if(err != CL_SUCCESS)
		throw CLException(err, __LINE__, __FILE__, "[CLBuffer]: Write failed.");
	std::memcpy(bufptr, data, _length);
	CL_EX(clEnqueueUnmapMemObject(m_cl_state->command_queue(), m_cl_memory, bufptr, 0u, nullptr, &unmap_event));
	return CLEvent{unmap_event};
}

ocl_template_matching::impl::cl::CLEvent ocl_template_matching::impl::cl::CLBuffer::buf_read(void* data, std::size_t length, std::size_t offset) const
{
	if(offset + length > m_size)
		throw std::out_of_range("[CLBuffer]: Buffer read failed. Input offset + length out of range.");
	if(m_flags | CL_MEM_HOST_WRITE_ONLY | CL_MEM_HOST_NO_ACCESS)
		throw std::runtime_error("[CLBuffer]: Reading from a write only buffer is not allowed.");
	std::size_t _offset = (length > 0ull ? offset : 0ull);
	std::size_t _length = (length > 0ull ? length : m_size);
	cl_int err{CL_SUCCESS};
	cl_event unmap_event{nullptr};
	void* bufptr = clEnqueueMapBuffer(m_cl_state->command_queue(), m_cl_memory, true, CL_MAP_READ, _offset, _length, static_cast<cl_uint>(m_event_cache.size()), (m_event_cache.size() > 0ull ? m_event_cache.data() : nullptr), nullptr, &err);
	if(err != CL_SUCCESS)
		throw CLException(err, __LINE__, __FILE__, "[CLBuffer]: Read failed.");
	std::memcpy(data, bufptr, _length);
	CL_EX(clEnqueueUnmapMemObject(m_cl_state->command_queue(), m_cl_memory, bufptr, 0u, nullptr, &unmap_event));
	return CLEvent{unmap_event};
}

void* ocl_template_matching::impl::cl::CLBuffer::map_buffer(std::size_t length, std::size_t offset, bool write, bool invalidate)
{
	cl_int err{CL_SUCCESS};
	void* bufptr = clEnqueueMapBuffer(m_cl_state->command_queue(), m_cl_memory, true, (write ? (invalidate ? CL_MAP_WRITE_INVALIDATE_REGION : CL_MAP_WRITE) : CL_MAP_READ), offset, length, static_cast<cl_uint>(m_event_cache.size()), (m_event_cache.size() > 0ull ? m_event_cache.data() : nullptr), nullptr, &err);
	if(err != CL_SUCCESS)
		throw CLException(err, __LINE__, __FILE__, "[CLBuffer]: Mapping buffer failed.");
	return bufptr;
}

ocl_template_matching::impl::cl::CLEvent ocl_template_matching::impl::cl::CLBuffer::unmap_buffer(void* bufptr)
{
	cl_event unmap_event{nullptr};
	CL_EX(clEnqueueUnmapMemObject(m_cl_state->command_queue(), m_cl_memory, bufptr, 0u, nullptr, &unmap_event));
	return CLEvent{unmap_event};
}

std::size_t ocl_template_matching::impl::cl::CLBuffer::size() const noexcept
{
	return m_size;
}

#pragma endregion

#pragma region class CLImage
// class CLImage

// predefined channel orders for the opencl order types
static constexpr ocl_template_matching::impl::cl::CLImage::HostChannelOrder HOST_CHANNEL_ORDER_R
{
	1ull,
	{
		ocl_template_matching::impl::cl::CLImage::HostChannel::R,
		ocl_template_matching::impl::cl::CLImage::HostChannel::R,
		ocl_template_matching::impl::cl::CLImage::HostChannel::R,
		ocl_template_matching::impl::cl::CLImage::HostChannel::R
	}
};

static constexpr ocl_template_matching::impl::cl::CLImage::HostChannelOrder HOST_CHANNEL_ORDER_RG
{
	1ull,
	{
		ocl_template_matching::impl::cl::CLImage::HostChannel::R,
		ocl_template_matching::impl::cl::CLImage::HostChannel::G,
		ocl_template_matching::impl::cl::CLImage::HostChannel::R,
		ocl_template_matching::impl::cl::CLImage::HostChannel::R
	}
};


static constexpr ocl_template_matching::impl::cl::CLImage::HostChannelOrder HOST_CHANNEL_ORDER_RGBA
{
	1ull,
	{
		ocl_template_matching::impl::cl::CLImage::HostChannel::R,
		ocl_template_matching::impl::cl::CLImage::HostChannel::G,
		ocl_template_matching::impl::cl::CLImage::HostChannel::B,
		ocl_template_matching::impl::cl::CLImage::HostChannel::A
	}
};


static constexpr ocl_template_matching::impl::cl::CLImage::HostChannelOrder HOST_CHANNEL_ORDER_BGRA
{
	1ull,
	{
		ocl_template_matching::impl::cl::CLImage::HostChannel::B,
		ocl_template_matching::impl::cl::CLImage::HostChannel::G,
		ocl_template_matching::impl::cl::CLImage::HostChannel::R,
		ocl_template_matching::impl::cl::CLImage::HostChannel::A
	}
};

// retrieve data type size of channel types
static std::size_t get_image_channel_type_size(const ocl_template_matching::impl::cl::CLImage::ImageChannelType& type)
{
	return std::size_t(static_cast<uint64_t>(type) & uint64_t { 0x00000000FFFFFFFF });
}

static std::size_t get_host_channel_type_size(const ocl_template_matching::impl::cl::CLImage::HostDataType& type)
{
	return std::size_t(static_cast<uint16_t>(type) & uint16_t { 0x00FF });
}

static std::size_t get_num_image_pixel_components(const ocl_template_matching::impl::cl::CLImage::ImageChannelOrder& channel_order)
{
	return std::size_t((static_cast<uint64_t>(channel_order) >> 24) & uint64_t { 0x00000000000000FF });
}

static std::size_t get_num_host_pixel_components(const ocl_template_matching::impl::cl::CLImage::HostChannelOrder& channel_order)
{
	return channel_order.num_channels;
}

static cl_uint get_image_channel_order_specifier(const ocl_template_matching::impl::cl::CLImage::ImageChannelOrder channel_order)
{
	return static_cast<cl_uint>((static_cast<uint64_t>(channel_order) >> 32) & uint64_t { 0x00000000FFFFFFFF });
}

static cl_uint get_image_channel_type_specifier(const ocl_template_matching::impl::cl::CLImage::ImageChannelType channel_type)
{
	return static_cast<cl_uint>((static_cast<uint64_t>(channel_type) >> 32) & uint64_t { 0x00000000FFFFFFFF });
}

ocl_template_matching::impl::cl::CLImage::CLImage(const std::shared_ptr<CLState>& clstate, const ImageDesc& image_desc) :
	m_image{nullptr},
	m_image_desc{image_desc},
	m_event_cache{},
	m_cl_state{clstate}
{
	cl_image_format fmt{get_image_channel_order_specifier(m_image_desc.channel_order), get_image_channel_type_specifier(m_image_desc.channel_type)};
	cl_image_desc desc{
		static_cast<cl_mem_object_type>(m_image_desc.type),
		m_image_desc.dimensions.width,
		m_image_desc.dimensions.height,
		m_image_desc.dimensions.depth,
		m_image_desc.dimensions.depth,
		0,
		0,
		0ull,
		0ull,
		nullptr
	};

	cl_int err{CL_SUCCESS};
	m_image = clCreateImage(m_cl_state->context(), static_cast<cl_mem_flags>(m_image_desc.access), &fmt, &desc, nullptr, &err);
	if(err != CL_SUCCESS)
		throw CLException(err, __LINE__, __FILE__, "[CLImage]: clCreateImage failed.");
}

ocl_template_matching::impl::cl::CLImage::~CLImage() noexcept
{
	if(m_image)
		CL(clReleaseMemObject(m_image));
}

ocl_template_matching::impl::cl::CLImage::CLImage(CLImage&& other) noexcept :
	m_image{other.m_image},
	m_image_desc{other.m_image_desc},
	m_event_cache{}
{
	other.m_image = nullptr;
}

ocl_template_matching::impl::cl::CLImage& ocl_template_matching::impl::cl::CLImage::operator=(CLImage&& other) noexcept
{
	if(this == &other)
		return *this;

	std::swap(m_image, other.m_image);
	std::swap(m_image_desc, other.m_image_desc);

	return *this;
}

std::size_t ocl_template_matching::impl::cl::CLImage::width() const
{
	return std::size_t{m_image_desc.dimensions.width};
}

std::size_t ocl_template_matching::impl::cl::CLImage::height() const
{
	return std::size_t{m_image_desc.dimensions.height};
}

std::size_t ocl_template_matching::impl::cl::CLImage::depth() const
{
	return std::size_t{m_image_desc.dimensions.depth};
}

std::size_t ocl_template_matching::impl::cl::CLImage::layers() const
{
	return std::size_t{m_image_desc.dimensions.depth};
}

bool ocl_template_matching::impl::cl::CLImage::match_format(const HostFormat& format)
{
	bool match{false};
	// check channel format
	switch(format.channel_type)
	{
	
		case HostDataType::INT8:
			match = (m_image_desc.channel_type == ImageChannelType::SNORM_INT8 || m_image_desc.channel_type == ImageChannelType::INT8);
			break;
		case HostDataType::INT16:
			match = (m_image_desc.channel_type == ImageChannelType::SNORM_INT16 || m_image_desc.channel_type == ImageChannelType::INT16);
			break;
		case HostDataType::INT32:
			match = (m_image_desc.channel_type == ImageChannelType::INT32);
			break;
		case HostDataType::UINT8:
			match = (m_image_desc.channel_type == ImageChannelType::UNORM_INT8 || m_image_desc.channel_type == ImageChannelType::UINT8);
			break;
		case HostDataType::UINT16:
			match = (m_image_desc.channel_type == ImageChannelType::UNORM_INT16 || m_image_desc.channel_type == ImageChannelType::UINT16);
			break;
		case HostDataType::UINT32:
			match = (m_image_desc.channel_type == ImageChannelType::UINT32);
			break;
		case HostDataType::HALF:
			match = (m_image_desc.channel_type == ImageChannelType::HALF);
			break;
		case HostDataType::FLOAT:
			match = (m_image_desc.channel_type == ImageChannelType::FLOAT);
			break;
		default:
			return false;
			break;
	}
	if(!match) return false;
	// check channel order
	match = false;
	switch(m_image_desc.channel_order)
	{
		case ImageChannelOrder::R:
			match = format.channel_order == HOST_CHANNEL_ORDER_R;
			break;
		case ImageChannelOrder::RG:
			match = format.channel_order == HOST_CHANNEL_ORDER_RG;
			break;
		case ImageChannelOrder::RGBA:
			match = format.channel_order == HOST_CHANNEL_ORDER_RGBA;
			break;
		case ImageChannelOrder::BGRA:
			match = format.channel_order == HOST_CHANNEL_ORDER_BGRA;
			break;
		case ImageChannelOrder::sRGBA:
			match = format.channel_order == HOST_CHANNEL_ORDER_RGBA;
			break;
		default:
			return false;
			break;
	}
	return match;
}

ocl_template_matching::impl::cl::CLEvent ocl_template_matching::impl::cl::CLImage::img_write(const HostFormat& format, const void* data_ptr, bool invalidate, ChannelDefaultValue default_value)
{
	if(!(format.im_region.dimensions.width && format.im_region.dimensions.height && format.im_region.dimensions.depth))
		throw std::runtime_error("[CLImage]: Write failed, region is empty.");
	// check if region matches
	if(	(format.im_region.offset.offset_width + format.im_region.dimensions.width > m_image_desc.dimensions.width)		||
		(format.im_region.offset.offset_height + format.im_region.dimensions.height > m_image_desc.dimensions.height)	||
		(format.im_region.offset.offset_depth + format.im_region.dimensions.depth > m_image_desc.dimensions.depth))
		throw std::runtime_error("[CLImage]: Write failed. Input region exceeds image dimensions.");
	// handle wrong pitch values
	if((m_image_desc.type == ImageType::Image1D || m_image_desc.type == ImageType::Image2D) && format.im_region.pitch.slice_pitch != 0ull)
		throw std::runtime_error("[CLImage]: Slice pitch must be 0 for 1D or 2D images.");

	// for parameterization of clEnqueueMapImage
	std::size_t origin[]{format.im_region.offset.offset_width, format.im_region.offset.offset_height, format.im_region.offset.offset_depth};
	std::size_t region[]{format.im_region.dimensions.width, format.im_region.dimensions.height, format.im_region.dimensions.depth};

	// pixel sizes for cl and host
	std::size_t cl_component_size = get_image_channel_type_size(m_image_desc.channel_type);
	std::size_t host_component_size = get_host_channel_type_size(format.channel_type);
	std::size_t cl_num_components = get_num_image_pixel_components(m_image_desc.channel_order);
	std::size_t host_num_components = get_num_host_pixel_components(format.channel_order);
	std::size_t cl_pixel_size = cl_component_size * cl_num_components;
	std::size_t host_pixel_size = host_component_size * host_num_components;

	// pitches for host in bytes
	std::size_t host_row_pitch = (format.im_region.pitch.row_pitch != 0ull ? format.im_region.pitch.row_pitch : format.im_region.dimensions.width * host_pixel_size);
	if(host_row_pitch < format.im_region.dimensions.width * host_pixel_size)
		throw std::runtime_error("[CLImage]: Row pitch must be >= region width * bytes per pixel.");
	std::size_t host_slice_pitch = (format.im_region.pitch.slice_pitch != 0ull ? format.im_region.pitch.slice_pitch : format.im_region.dimensions.height * host_row_pitch);
	if(host_slice_pitch < format.im_region.dimensions.height * host_row_pitch)
		throw std::runtime_error("[CLImage]: Row pitch must be >= height * host row pitch.");

	// map image region
	cl_int err{CL_SUCCESS};
	cl_event map_event;
	std::size_t row_pitch{0ull};
	std::size_t slice_pitch{0ull};
	// cast mapped pointer to uint8_t. This way we are allowed to do byte-wise pointer arithmetic.
	uint8_t* img_ptr = static_cast<uint8_t*>(clEnqueueMapImage(
		m_cl_state->command_queue(),
		m_image,
		CL_TRUE,
		(invalidate ? CL_MAP_WRITE_INVALIDATE_REGION : CL_MAP_WRITE),
		&origin[0],
		&region[0],
		&row_pitch,
		&slice_pitch,
		static_cast<cl_uint>(m_event_cache.size()),
		(m_event_cache.size() > 0ull ? m_event_cache.data() : nullptr),
		nullptr,
		&err
	));
	if(err != CL_SUCCESS)
		throw CLException(err, __LINE__, __FILE__, "[CLImage]: clEnqueueMapImage failed.");

	// if slice_pitch is 0 we have a 1D o 2D image. Re-use slice_pitch in this case:
	slice_pitch = slice_pitch ? slice_pitch : row_pitch * format.im_region.dimensions.height;
	// determine size of copied memory regions
	std::size_t row_size = std::min(row_pitch, host_row_pitch);
	std::size_t slice_size = std::min(slice_pitch, host_slice_pitch);
	std::size_t region_size = format.im_region.dimensions.depth * host_slice_pitch;

	// host format must match image format
	if(match_format(format))
	{
		if(host_slice_pitch == slice_pitch) // we can copy the whole region at once
		{
			std::memcpy(img_ptr, data_ptr, region_size);
		}
		else // we have to copy slices separately
		{
			if(host_row_pitch == row_pitch) // we can copy whole slices at once
			{
				uint8_t* cur_img_ptr = img_ptr;
				const uint8_t* cur_data_ptr = static_cast<const uint8_t*>(data_ptr);
				for(std::size_t slice_idx = 0; slice_idx < format.im_region.dimensions.depth; ++slice_idx) // copy one slice at a time
				{
					std::memcpy(cur_img_ptr, cur_data_ptr, slice_size);
					cur_img_ptr += slice_pitch;
					cur_data_ptr += host_slice_pitch;
				}
			}
			else // we have to copy row-by-row
			{
				uint8_t* cur_img_ptr = img_ptr;
				const uint8_t* cur_data_ptr = static_cast<const uint8_t*>(data_ptr);
				for(std::size_t slice_idx = 0; slice_idx < format.im_region.dimensions.depth; ++slice_idx)
				{
					uint8_t* cur_row_img_ptr = cur_img_ptr;
					const uint8_t* cur_row_data_ptr = cur_data_ptr;
					for(std::size_t row_idx = 0; row_idx < format.im_region.dimensions.height; ++row_idx) // copy row by row
					{
						std::memcpy(cur_row_img_ptr, cur_row_data_ptr, row_size);
						cur_row_img_ptr += row_pitch;
						cur_row_data_ptr += host_row_pitch;
					}
					cur_img_ptr += slice_pitch;
					cur_data_ptr += host_slice_pitch;
				}
			}
		}
	}
	else
		throw std::runtime_error("[CLImage]: Image write failed. Host format does not match image format.");

	// unmap image and return event
	CL_EX(clEnqueueUnmapMemObject(m_cl_state->command_queue(), m_image, img_ptr, 0ull, nullptr, &map_event));
	return map_event;
}

ocl_template_matching::impl::cl::CLEvent ocl_template_matching::impl::cl::CLImage::img_read(const HostFormat& format, void* data_ptr, ChannelDefaultValue default_value)
{
	if(!(format.im_region.dimensions.width && format.im_region.dimensions.height && format.im_region.dimensions.depth))
		throw std::runtime_error("[CLImage]: Read failed, region is empty.");
	// check if region matches
	if((format.im_region.offset.offset_width + format.im_region.dimensions.width > m_image_desc.dimensions.width) ||
		(format.im_region.offset.offset_height + format.im_region.dimensions.height > m_image_desc.dimensions.height) ||
		(format.im_region.offset.offset_depth + format.im_region.dimensions.depth > m_image_desc.dimensions.depth))
		throw std::runtime_error("[CLImage]: Read failed. Input region exceeds image dimensions.");
	// handle wrong pitch values
	if((m_image_desc.type == ImageType::Image1D || m_image_desc.type == ImageType::Image2D) && format.im_region.pitch.slice_pitch != 0ull)
		throw std::runtime_error("[CLImage]: Slice pitch must be 0 for 1D or 2D images.");

	// for parameterization of clEnqueueMapImage
	std::size_t origin[]{format.im_region.offset.offset_width, format.im_region.offset.offset_height, format.im_region.offset.offset_depth};
	std::size_t region[]{format.im_region.dimensions.width, format.im_region.dimensions.height, format.im_region.dimensions.depth};

	// pixel sizes for cl and host
	std::size_t cl_component_size = get_image_channel_type_size(m_image_desc.channel_type);
	std::size_t host_component_size = get_host_channel_type_size(format.channel_type);
	std::size_t cl_num_components = get_num_image_pixel_components(m_image_desc.channel_order);
	std::size_t host_num_components = get_num_host_pixel_components(format.channel_order);
	std::size_t cl_pixel_size = cl_component_size * cl_num_components;
	std::size_t host_pixel_size = host_component_size * host_num_components;

	// pitches for host in bytes
	std::size_t host_row_pitch = (format.im_region.pitch.row_pitch != 0ull ? format.im_region.pitch.row_pitch : format.im_region.dimensions.width * host_pixel_size);
	if(host_row_pitch < format.im_region.dimensions.width * host_pixel_size)
		throw std::runtime_error("[CLImage]: Row pitch must be >= region width * bytes per pixel.");
	std::size_t host_slice_pitch = (format.im_region.pitch.slice_pitch != 0ull ? format.im_region.pitch.slice_pitch : format.im_region.dimensions.height * host_row_pitch);
	if(host_slice_pitch < format.im_region.dimensions.height * host_row_pitch)
		throw std::runtime_error("[CLImage]: Row pitch must be >= height * host row pitch.");

	// map image region
	cl_int err{CL_SUCCESS};
	cl_event map_event;
	std::size_t row_pitch{0ull};
	std::size_t slice_pitch{0ull};
	// cast mapped pointer to uint8_t. This way we are allowed to do byte-wise pointer arithmetic.
	uint8_t* img_ptr = static_cast<uint8_t*>(clEnqueueMapImage(
		m_cl_state->command_queue(),
		m_image,
		CL_TRUE,
		CL_MAP_READ,
		&origin[0],
		&region[0],
		&row_pitch,
		&slice_pitch,
		static_cast<cl_uint>(m_event_cache.size()),
		(m_event_cache.size() > 0ull ? m_event_cache.data() : nullptr),
		nullptr,
		&err
	));
	if(err != CL_SUCCESS)
		throw CLException(err, __LINE__, __FILE__, "[CLImage]: clEnqueueMapImage failed.");

	// if slice_pitch is 0 we have a 1D o 2D image. Re-use slice_pitch in this case:
	slice_pitch = slice_pitch ? slice_pitch : row_pitch * format.im_region.dimensions.height;
	// determine size of copied memory regions
	std::size_t row_size = std::min(row_pitch, host_row_pitch);
	std::size_t slice_size = std::min(slice_pitch, host_slice_pitch);
	std::size_t region_size = format.im_region.dimensions.depth * host_slice_pitch;

	// host format must match image format
	if(match_format(format))
	{
		if(host_slice_pitch == slice_pitch) // we can copy the whole region at once
		{
			std::memcpy(data_ptr, img_ptr, region_size);
		}
		else // we have to copy slices separately
		{
			if(host_row_pitch == row_pitch) // we can copy whole slices at once
			{
				const uint8_t* cur_img_ptr = img_ptr;
				uint8_t* cur_data_ptr = static_cast<uint8_t*>(data_ptr);
				for(std::size_t slice_idx = 0; slice_idx < format.im_region.dimensions.depth; ++slice_idx) // copy one slice at a time
				{
					std::memcpy(cur_data_ptr, cur_img_ptr, slice_size);
					cur_img_ptr += slice_pitch;
					cur_data_ptr += host_slice_pitch;
				}
			}
			else // we have to copy row-by-row
			{
				const uint8_t* cur_img_ptr = img_ptr;
				uint8_t* cur_data_ptr = static_cast<uint8_t*>(data_ptr);
				for(std::size_t slice_idx = 0; slice_idx < format.im_region.dimensions.depth; ++slice_idx)
				{
					const uint8_t* cur_row_img_ptr = cur_img_ptr;
					uint8_t* cur_row_data_ptr = cur_data_ptr;
					for(std::size_t row_idx = 0; row_idx < format.im_region.dimensions.height; ++row_idx) // copy row by row
					{
						std::memcpy(cur_row_data_ptr, cur_row_img_ptr, row_size);
						cur_row_img_ptr += row_pitch;
						cur_row_data_ptr += host_row_pitch;
					}
					cur_img_ptr += slice_pitch;
					cur_data_ptr += host_slice_pitch;
				}
			}
		}
	}
	else
		throw std::runtime_error("[CLImage]: Image read failed. Host format does not match image format.");

	// unmap image and return event
	CL_EX(clEnqueueUnmapMemObject(m_cl_state->command_queue(), m_image, img_ptr, 0ull, nullptr, &map_event));
	return map_event;
}

#pragma endregion
#pragma endregion
