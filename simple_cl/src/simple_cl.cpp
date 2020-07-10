#include <simple_cl.hpp>

// -------------------------------------------- NAMESPACE simple_cl::util-----------------------------------
#pragma region util

std::vector<std::string> simple_cl::util::string_split(const std::string& s, char delimiter)
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

unsigned int simple_cl::util::get_cl_version_num(const std::string& str)
{
	std::string version_string = util::string_split(str, ' ')[1];
	unsigned int version_major = static_cast<unsigned int>(std::stoul(util::string_split(version_string, '.')[0]));
	unsigned int version_minor = static_cast<unsigned int>(std::stoul(util::string_split(version_string, '.')[1]));
	return version_major * 100u + version_minor * 10u;
}

#pragma endregion

// -------------------------------------------- NAMESPACE simple_cl::cl -------------------------------------

#pragma region cl
#pragma region class Context
// ---------------------- class Context
// factory function
std::shared_ptr<simple_cl::cl::Context> simple_cl::cl::Context::createInstance(std::size_t platform_index, std::size_t device_index)
{
	return std::shared_ptr<Context>(new Context{platform_index, device_index});
}

simple_cl::cl::Context::Context(std::size_t platform_index, std::size_t device_index) :
	m_available_platforms{std::move(read_platform_and_device_info())},
	m_selected_platform_index{0},
	m_selected_device_index{0},
	m_context{nullptr},
	m_command_queue{nullptr},
	m_cl_ex_holder{nullptr}		
{
	try
	{
		init_cl_instance(platform_index, device_index);
	}
	catch(...)
	{
		cleanup();
		std::cerr << "[OCL_TEMPLATE_MATCHER]: OpenCL initialization failed." << std::endl;
		throw;
	}
}

simple_cl::cl::Context::~Context()
{
	cleanup();
}

simple_cl::cl::Context::Context(Context&& other) noexcept :
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

simple_cl::cl::Context& simple_cl::cl::Context::operator=(Context&& other) noexcept
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

void simple_cl::cl::Context::print_selected_platform_info() const
{
	std::cout << "===== Selected OpenCL platform =====" << std::endl;
	std::cout << m_available_platforms[m_selected_platform_index];
}

void simple_cl::cl::Context::print_selected_device_info() const
{
	std::cout << "===== Selected OpenCL device =====" << std::endl;
	std::cout << m_available_platforms[m_selected_platform_index].devices[m_selected_device_index];
}

void simple_cl::cl::Context::print_platform_and_device_info(const std::vector<Context::CLPlatform>& available_platforms)
{
	std::cout << "===== SUITABLE OpenCL PLATFORMS AND DEVICES =====" << std::endl;
	for(std::size_t p = 0ull; p < available_platforms.size(); ++p)
	{
		std::cout << "[Platform ID: " << p << "] " << available_platforms[p] << std::endl;
		std::cout << "Suitable OpenCL 1.2+ devices:" << std::endl;
		for(std::size_t d = 0ull; d < available_platforms[p].devices.size(); ++d)
		{
			std::cout << std::endl;
			std::cout << "[Platform ID: " << p << "]" << "[Device ID: " << d << "] " << available_platforms[p].devices[d];
		}
	}
}

void simple_cl::cl::Context::print_platform_and_device_info()
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

std::vector<simple_cl::cl::Context::CLPlatform> simple_cl::cl::Context::read_platform_and_device_info()
{
	// output vector
	std::vector<CLPlatform> available_platforms;
	// query number of platforms available
	cl_uint number_of_platforms{0};
	CL_EX(clGetPlatformIDs(0, nullptr, &number_of_platforms));
	if(number_of_platforms == 0u)
		return available_platforms;
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
				available_platforms.push_back(std::move(platform));
			}
		}
	}
	return available_platforms;
}

void simple_cl::cl::Context::init_cl_instance(std::size_t platform_id, std::size_t device_id)
{
	if(m_available_platforms.size() == 0ull)
		throw std::runtime_error("[OCL_TEMPLATE_MATCHER]: No suitable OpenCL 1.2 platform found.");
	if(platform_id >= m_available_platforms.size())
		throw std::runtime_error("[OCL_TEMPLATE_MATCHER]: Platform index out of range.");
	if(m_available_platforms[platform_id].devices.size() == 0ull)
		throw std::runtime_error("[OCL_TEMPLATE_MATCHER]: No suitable OpenCL 1.2 device found.");
	if(device_id >= m_available_platforms[platform_id].devices.size())
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

void simple_cl::cl::Context::cleanup()
{
	if(m_command_queue)
		CL(clReleaseCommandQueue(m_command_queue));
	m_command_queue = nullptr;
	if(m_context)
		CL(clReleaseContext(m_context));
	m_context = nullptr;
	m_cl_ex_holder.ex_msg = nullptr;
}

const simple_cl::cl::Context::CLPlatform& simple_cl::cl::Context::get_selected_platform() const
{
	return m_available_platforms[m_selected_platform_index];
}

const simple_cl::cl::Context::CLDevice& simple_cl::cl::Context::get_selected_device() const
{
	return m_available_platforms[m_selected_platform_index].devices[m_selected_device_index];
}

std::ostream& simple_cl::cl::operator<<(std::ostream& os, const simple_cl::cl::Context::CLPlatform& plat)
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

std::ostream& simple_cl::cl::operator<<(std::ostream& os, const simple_cl::cl::Context::CLDevice& dev)
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

void simple_cl::cl::create_context_callback(const char* errinfo, const void* private_info, std::size_t cb, void* user_data)
{
	static_cast<Context::CLExHolder*>(user_data)->ex_msg = errinfo;
}

#pragma endregion

#pragma region class Program
// -------------------------- class Program

simple_cl::cl::Program::Program(const std::string& kernel_source, const std::string& compiler_options, const std::shared_ptr<Context>& clstate) :
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
			m_kernels[kernel_names[i]] = CLKernel{i, {}, kernel};
			// query per-kernel info
			CLKernelInfo kinfo;
			std::size_t sz{0ull};
			cl_ulong usz{0ul};
			CL_EX(clGetKernelWorkGroupInfo(kernel, m_cl_state->get_selected_device().device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(std::size_t), &sz, nullptr));
			kinfo.max_work_group_size = sz; sz = 0ull;
			CL_EX(clGetKernelWorkGroupInfo(kernel, m_cl_state->get_selected_device().device_id, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(cl_ulong), &usz, nullptr));
			kinfo.local_memory_usage = std::size_t{usz}; usz = 0ul;
			CL_EX(clGetKernelWorkGroupInfo(kernel, m_cl_state->get_selected_device().device_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(std::size_t), &sz, nullptr));
			kinfo.preferred_work_group_size_multiple = sz; sz = 0ull;
			CL_EX(clGetKernelWorkGroupInfo(kernel, m_cl_state->get_selected_device().device_id, CL_KERNEL_PRIVATE_MEM_SIZE, sizeof(cl_ulong), &usz, nullptr));
			kinfo.private_memory_usage = std::size_t{usz};
			m_kernels[kernel_names[i]].kernel_info = kinfo;
		}
	}
	catch(...)
	{
		cleanup();
		throw;
	}
}

simple_cl::cl::Program::~Program()
{
	cleanup();
}

simple_cl::cl::Program::Program(Program&& other) noexcept :
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

simple_cl::cl::Program& simple_cl::cl::Program::operator=(Program&& other) noexcept
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

void simple_cl::cl::Program::cleanup() noexcept
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

simple_cl::cl::Event simple_cl::cl::Program::invoke(cl_kernel kernel, const std::vector<cl_event>& dep_events, const ExecParams& exparams)
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
	return Event{ev};
}

void simple_cl::cl::Program::setKernelArgsImpl(const std::string& name, std::size_t index, std::size_t arg_size, const void* arg_data_ptr)
{
	try
	{
		CL_EX(clSetKernelArg(m_kernels.at(name).kernel, static_cast<cl_uint>(index), arg_size, arg_data_ptr));
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

void simple_cl::cl::Program::setKernelArgsImpl(cl_kernel kernel, std::size_t index, std::size_t arg_size, const void* arg_data_ptr)
{	
	CL_EX(clSetKernelArg(kernel, static_cast<cl_uint>(index), arg_size, arg_data_ptr));
}

simple_cl::cl::Program::CLKernelHandle simple_cl::cl::Program::getKernel(const std::string& name) const
{
	try
	{
		const auto& kernel{m_kernels.at(name)};
		return CLKernelHandle{kernel.kernel, kernel.kernel_info};
	}
	catch(const std::out_of_range&)
	{
		throw std::runtime_error("Unknown kernel name.");
	}
}

simple_cl::cl::Program::CLKernelInfo simple_cl::cl::Program::getKernelInfo(const std::string& name) const
{
	try
	{
		return m_kernels.at(name).kernel_info;
	}
	catch(const std::out_of_range&)
	{
		throw std::runtime_error("Unknown kernel name.");
	}
}

simple_cl::cl::Program::CLKernelInfo simple_cl::cl::Program::getKernelInfo(const CLKernelHandle& kernel) const
{
	assert(kernel.m_kernel);
	return kernel.m_kernel_info;
}

#pragma endregion

#pragma region class Event
// class Event
simple_cl::cl::Event::Event(cl_event ev) :
	m_event{ev}
{	
}

simple_cl::cl::Event::~Event()
{
	if(m_event)
		CL_EX(clReleaseEvent(m_event));
}

simple_cl::cl::Event::Event(const Event& other) :
	m_event{other.m_event}
{
	if(m_event)
		CL_EX(clRetainEvent(m_event));
}

simple_cl::cl::Event::Event(Event&& other) noexcept :
	m_event{other.m_event}
{
	other.m_event = nullptr;
}

simple_cl::cl::Event& simple_cl::cl::Event::operator=(const Event& other)
{
	if(this == &other)
		return *this;

	m_event = other.m_event;
	CL_EX(clRetainEvent(m_event));

	return *this;
}

simple_cl::cl::Event& simple_cl::cl::Event::operator=(Event&& other) noexcept
{
	if(this == &other)
		return *this;

	std::swap(m_event, other.m_event);

	return *this;
}

void simple_cl::cl::Event::wait() const
{
	CL_EX(clWaitForEvents(1, &m_event));
}

void simple_cl::cl::Event::wait_for_events_(const std::vector<cl_event>& events)
{
	CL_EX(clWaitForEvents(static_cast<cl_uint>(events.size()), events.data()));
}
#pragma endregion

#pragma region class Buffer
// class Buffer

simple_cl::cl::Buffer::Buffer(std::size_t size, const MemoryFlags& flags, const std::shared_ptr<Context>& clstate, void* hostptr) :
	m_cl_memory{nullptr},
	m_size{0ull},
	m_cl_state{clstate},
	m_flags{flags},
	m_hostptr{nullptr},
	m_event_cache{}
{	
	cl_int err{CL_SUCCESS};
	cl_mem_flags clflags{static_cast<cl_mem_flags>(flags.device_access) | static_cast<cl_mem_flags>(flags.host_access) | static_cast<cl_mem_flags>(flags.host_pointer_option)};
	m_cl_memory = clCreateBuffer(m_cl_state->context(), clflags, size, (flags.host_pointer_option == HostPointerOption::UseHostPtr || flags.host_pointer_option == HostPointerOption::CopyHostPtr) ? hostptr : nullptr, &err);
	if(err != CL_SUCCESS)
		throw CLException(err, __LINE__, __FILE__, "[Buffer]: OpenCL buffer creation failed.");
	m_size = size;
	m_flags = flags;
	m_hostptr = (flags.host_pointer_option == HostPointerOption::UseHostPtr || flags.host_pointer_option == HostPointerOption::CopyHostPtr) ? hostptr : nullptr;
}

simple_cl::cl::Buffer::~Buffer() noexcept
{
	if(m_cl_memory)
		CL(clReleaseMemObject(m_cl_memory));
}

simple_cl::cl::Buffer::Buffer(Buffer&& other) noexcept :
	m_cl_memory{nullptr},
	m_size{0ull},
	m_cl_state{nullptr},
	m_flags{},
	m_hostptr{nullptr},
	m_event_cache{}
{
	std::swap(m_cl_memory, other.m_cl_memory);
	std::swap(m_size, other.m_size);
	std::swap(m_cl_state, other.m_cl_state);
	std::swap(m_flags, other.m_flags);
	std::swap(m_hostptr, other.m_hostptr);
}

simple_cl::cl::Buffer& simple_cl::cl::Buffer::operator=(Buffer&& other) noexcept
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

simple_cl::cl::Event simple_cl::cl::Buffer::buf_write(const void* data, std::size_t length, std::size_t offset, bool invalidate)
{
	if(offset + length > m_size)
		throw std::out_of_range("[Buffer]: Buffer write failed. Input offset + length out of range.");
	if(m_flags.host_access == HostAccess::ReadOnly || m_flags.host_access == HostAccess::NoAccess)
		throw std::runtime_error("[Buffer]: Writing to a read only buffer is not allowed.");
	std::size_t _offset = (length > 0ull ? offset : 0ull);
	std::size_t _length = (length > 0ull ? length : m_size);
	cl_int err{CL_SUCCESS};
	cl_event unmap_event{nullptr};
	void* bufptr = clEnqueueMapBuffer(m_cl_state->command_queue(), m_cl_memory, true, (invalidate ? CL_MAP_WRITE_INVALIDATE_REGION : CL_MAP_WRITE), _offset, _length, static_cast<cl_uint>(m_event_cache.size()), (m_event_cache.size() > 0ull? m_event_cache.data() : nullptr), nullptr, &err);
	if(err != CL_SUCCESS)
		throw CLException(err, __LINE__, __FILE__, "[Buffer]: Write failed.");
	std::memcpy(bufptr, data, _length);
	CL_EX(clEnqueueUnmapMemObject(m_cl_state->command_queue(), m_cl_memory, bufptr, 0u, nullptr, &unmap_event));
	return Event{unmap_event};
}

simple_cl::cl::Event simple_cl::cl::Buffer::buf_read(void* data, std::size_t length, std::size_t offset) const
{
	if(offset + length > m_size)
		throw std::out_of_range("[Buffer]: Buffer read failed. Input offset + length out of range.");
	if(m_flags.host_access == HostAccess::WriteOnly || m_flags.host_access == HostAccess::NoAccess)
		throw std::runtime_error("[Buffer]: Reading from a write only buffer is not allowed.");
	std::size_t _offset = (length > 0ull ? offset : 0ull);
	std::size_t _length = (length > 0ull ? length : m_size);
	cl_int err{CL_SUCCESS};
	cl_event unmap_event{nullptr};
	void* bufptr = clEnqueueMapBuffer(m_cl_state->command_queue(), m_cl_memory, true, CL_MAP_READ, _offset, _length, static_cast<cl_uint>(m_event_cache.size()), (m_event_cache.size() > 0ull ? m_event_cache.data() : nullptr), nullptr, &err);
	if(err != CL_SUCCESS)
		throw CLException(err, __LINE__, __FILE__, "[Buffer]: Read failed.");
	std::memcpy(data, bufptr, _length);
	CL_EX(clEnqueueUnmapMemObject(m_cl_state->command_queue(), m_cl_memory, bufptr, 0u, nullptr, &unmap_event));
	return Event{unmap_event};
}

void* simple_cl::cl::Buffer::map_buffer(std::size_t length, std::size_t offset, bool write, bool invalidate)
{
	cl_int err{CL_SUCCESS};
	void* bufptr = clEnqueueMapBuffer(m_cl_state->command_queue(), m_cl_memory, true, (write ? (invalidate ? CL_MAP_WRITE_INVALIDATE_REGION : CL_MAP_WRITE) : CL_MAP_READ), offset, length, static_cast<cl_uint>(m_event_cache.size()), (m_event_cache.size() > 0ull ? m_event_cache.data() : nullptr), nullptr, &err);
	if(err != CL_SUCCESS)
		throw CLException(err, __LINE__, __FILE__, "[Buffer]: Mapping buffer failed.");
	return bufptr;
}

simple_cl::cl::Event simple_cl::cl::Buffer::unmap_buffer(void* bufptr)
{
	cl_event unmap_event{nullptr};
	CL_EX(clEnqueueUnmapMemObject(m_cl_state->command_queue(), m_cl_memory, bufptr, 0u, nullptr, &unmap_event));
	return Event{unmap_event};
}

std::size_t simple_cl::cl::Buffer::size() const noexcept
{
	return m_size;
}

#pragma endregion

#pragma region class Image
// class Image

simple_cl::cl::Image::Image(const std::shared_ptr<Context>& clstate, const ImageDesc& image_desc) :
	m_image{nullptr},
	m_image_desc{image_desc},
	m_event_cache{},
	m_cl_state{clstate}
{
	m_image_desc.host_ptr = (image_desc.flags.host_pointer_option == HostPointerOption::UseHostPtr || image_desc.flags.host_pointer_option == HostPointerOption::CopyHostPtr) ? image_desc.host_ptr : nullptr;
	cl_image_format fmt{get_image_channel_order_specifier(m_image_desc.channel_order), get_image_channel_type_specifier(m_image_desc.channel_type)};
	cl_image_desc desc{
		static_cast<cl_mem_object_type>(m_image_desc.type),
		m_image_desc.dimensions.width,
		m_image_desc.dimensions.height,
		m_image_desc.type == ImageType::Image3D ? m_image_desc.dimensions.depth : 1ull,
		(m_image_desc.type == ImageType::Image1DArray || m_image_desc.type == ImageType::Image2DArray) ? m_image_desc.dimensions.depth : 1ull,
		m_image_desc.host_ptr ? m_image_desc.pitch.row_pitch : 0ull,
		m_image_desc.host_ptr ? m_image_desc.pitch.slice_pitch : 0ull,
		0ull,
		0ull,
		nullptr
	};

	cl_int err{CL_SUCCESS};
	cl_mem_flags clflags{static_cast<cl_mem_flags>(m_image_desc.flags.device_access) | static_cast<cl_mem_flags>(m_image_desc.flags.host_access) | static_cast<cl_mem_flags>(m_image_desc.flags.host_pointer_option)};
	m_image = clCreateImage(m_cl_state->context(), clflags, &fmt, &desc, m_image_desc.host_ptr, &err);
	if(err != CL_SUCCESS)
		throw CLException(err, __LINE__, __FILE__, "[Image]: clCreateImage failed.");
}

simple_cl::cl::Image::~Image() noexcept
{
	if(m_image)
		CL(clReleaseMemObject(m_image));
}

simple_cl::cl::Image::Image(Image&& other) noexcept :
	m_image{other.m_image},
	m_image_desc{other.m_image_desc},
	m_event_cache{}
{
	other.m_image = nullptr;
}

simple_cl::cl::Image& simple_cl::cl::Image::operator=(Image&& other) noexcept
{
	if(this == &other)
		return *this;

	std::swap(m_image, other.m_image);
	std::swap(m_image_desc, other.m_image_desc);

	return *this;
}

std::size_t simple_cl::cl::Image::width() const
{
	return std::size_t{m_image_desc.dimensions.width};
}

std::size_t simple_cl::cl::Image::height() const
{
	return std::size_t{m_image_desc.dimensions.height};
}

std::size_t simple_cl::cl::Image::depth() const
{
	return std::size_t{m_image_desc.dimensions.depth};
}

std::size_t simple_cl::cl::Image::layers() const
{
	return std::size_t{m_image_desc.dimensions.depth};
}

bool simple_cl::cl::Image::match_format(const HostFormat& format)
{
	// check channel data type
	if(!(get_host_channel_base_type(format.channel_type) == get_image_channel_base_type(m_image_desc.channel_type)))
		return false;
	// check channel order
	if(!(get_num_host_pixel_components(format.channel_order) == get_num_image_pixel_components(m_image_desc.channel_order)))
		return false;
	// iterate over host color channels and check if the corresponding image color channel matches
	for(std::size_t i = 0; i < format.channel_order.num_channels; ++i)
		if(format.channel_order.channels[i] != get_image_color_channel(m_image_desc.channel_order, i))
			return false;
	// success!
	return true;
}

simple_cl::cl::Event simple_cl::cl::Image::img_write_mapped(const ImageRegion& img_region, const HostFormat& format, const void* data_ptr, bool invalidate, ChannelDefaultValue default_value)
{
	if(m_image_desc.flags.host_access == HostAccess::NoAccess || m_image_desc.flags.host_access == HostAccess::ReadOnly)
		throw std::runtime_error("[Image]: Host is not allowed to write this image.");
	if(!(img_region.dimensions.width && img_region.dimensions.height && img_region.dimensions.depth))
		throw std::runtime_error("[Image]: Write failed, region is empty.");
	// check if region matches
	if(	(img_region.offset.offset_width + img_region.dimensions.width > m_image_desc.dimensions.width)		||
		(img_region.offset.offset_height + img_region.dimensions.height > m_image_desc.dimensions.height)	||
		(img_region.offset.offset_depth + img_region.dimensions.depth > m_image_desc.dimensions.depth))
		throw std::runtime_error("[Image]: Write failed. Input region exceeds image dimensions.");
	// handle wrong pitch values
	if((m_image_desc.type == ImageType::Image1D || m_image_desc.type == ImageType::Image2D) && format.pitch.slice_pitch != 0ull)
		throw std::runtime_error("[Image]: Slice pitch must be 0 for 1D or 2D images.");

	// for parameterization of clEnqueueMapImage
	std::size_t origin[]{img_region.offset.offset_width, img_region.offset.offset_height, img_region.offset.offset_depth};
	std::size_t region[]{img_region.dimensions.width, img_region.dimensions.height, img_region.dimensions.depth};

	// pixel sizes for cl and host
	std::size_t cl_component_size = get_image_channel_type_size(m_image_desc.channel_type);
	std::size_t host_component_size = get_host_channel_type_size(format.channel_type);
	std::size_t cl_num_components = get_num_image_pixel_components(m_image_desc.channel_order);
	std::size_t host_num_components = get_num_host_pixel_components(format.channel_order);
	std::size_t cl_pixel_size = cl_component_size * cl_num_components;
	std::size_t host_pixel_size = host_component_size * host_num_components;

	// pitches for host in bytes
	std::size_t host_row_pitch = (format.pitch.row_pitch != 0ull ? format.pitch.row_pitch : img_region.dimensions.width * host_pixel_size);
	if(host_row_pitch < img_region.dimensions.width * host_pixel_size)
		throw std::runtime_error("[Image]: Row pitch must be >= region width * bytes per pixel.");
	std::size_t host_slice_pitch = (format.pitch.slice_pitch != 0ull ? format.pitch.slice_pitch : img_region.dimensions.height * host_row_pitch);
	if(host_slice_pitch < img_region.dimensions.height * host_row_pitch)
		throw std::runtime_error("[Image]: Row pitch must be >= height * host row pitch.");

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
		throw CLException(err, __LINE__, __FILE__, "[Image]: clEnqueueMapImage failed.");

	// if slice_pitch is 0 we have a 1D o 2D image. Re-use slice_pitch in this case:
	slice_pitch = slice_pitch ? slice_pitch : row_pitch * img_region.dimensions.height;
	// determine size of copied memory regions
	std::size_t row_size = std::min(row_pitch, host_row_pitch);
	std::size_t slice_size = std::min(slice_pitch, host_slice_pitch);
	std::size_t region_size = img_region.dimensions.depth * host_slice_pitch;

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
				for(std::size_t slice_idx = 0; slice_idx < img_region.dimensions.depth; ++slice_idx) // copy one slice at a time
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
				for(std::size_t slice_idx = 0; slice_idx < img_region.dimensions.depth; ++slice_idx)
				{
					uint8_t* cur_row_img_ptr = cur_img_ptr;
					const uint8_t* cur_row_data_ptr = cur_data_ptr;
					for(std::size_t row_idx = 0; row_idx < img_region.dimensions.height; ++row_idx) // copy row by row
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
		throw std::runtime_error("[Image]: Image write failed. Host format does not match image format.");

	// unmap image and return event
	CL_EX(clEnqueueUnmapMemObject(m_cl_state->command_queue(), m_image, img_ptr, 0ull, nullptr, &map_event));
	return Event{map_event};
}

simple_cl::cl::Event simple_cl::cl::Image::img_write(const ImageRegion& img_region, const HostFormat& format, const void* data_ptr, bool blocking, ChannelDefaultValue default_value)
{
	if(m_image_desc.flags.host_access == HostAccess::NoAccess || m_image_desc.flags.host_access == HostAccess::ReadOnly)
		throw std::runtime_error("[Image]: Host is not allowed to write this image.");
	if(!(img_region.dimensions.width && img_region.dimensions.height && img_region.dimensions.depth))
		throw std::runtime_error("[Image]: Write failed, region is empty.");
	// check if region matches
	if((img_region.offset.offset_width + img_region.dimensions.width > m_image_desc.dimensions.width) ||
		(img_region.offset.offset_height + img_region.dimensions.height > m_image_desc.dimensions.height) ||
		(img_region.offset.offset_depth + img_region.dimensions.depth > m_image_desc.dimensions.depth))
		throw std::runtime_error("[Image]: Write failed. Input region exceeds image dimensions.");
	// handle wrong pitch values
	if((m_image_desc.type == ImageType::Image1D || m_image_desc.type == ImageType::Image2D) && format.pitch.slice_pitch != 0ull)
		throw std::runtime_error("[Image]: Slice pitch must be 0 for 1D or 2D images.");

	// ensure matching image format
	if(!match_format(format))
		throw std::runtime_error("[Image]: Write failed. Host format doesn't match image format.");

	// for parameterization of clEnqueueMapImage
	std::size_t origin[]{img_region.offset.offset_width, img_region.offset.offset_height, img_region.offset.offset_depth};
	std::size_t region[]{img_region.dimensions.width, img_region.dimensions.height, img_region.dimensions.depth};

	// pixel sizes for cl and host
	std::size_t cl_component_size = get_image_channel_type_size(m_image_desc.channel_type);
	std::size_t host_component_size = get_host_channel_type_size(format.channel_type);
	std::size_t cl_num_components = get_num_image_pixel_components(m_image_desc.channel_order);
	std::size_t host_num_components = get_num_host_pixel_components(format.channel_order);
	std::size_t cl_pixel_size = cl_component_size * cl_num_components;
	std::size_t host_pixel_size = host_component_size * host_num_components;

	// pitches for host in bytes
	std::size_t host_row_pitch = (format.pitch.row_pitch != 0ull ? format.pitch.row_pitch : img_region.dimensions.width * host_pixel_size);
	if(host_row_pitch < img_region.dimensions.width * host_pixel_size)
		throw std::runtime_error("[Image]: Row pitch must be >= region width * bytes per pixel.");
	std::size_t host_slice_pitch = (format.pitch.slice_pitch != 0ull ? format.pitch.slice_pitch : img_region.dimensions.height * host_row_pitch);
	if(host_slice_pitch < img_region.dimensions.height * host_row_pitch)
		throw std::runtime_error("[Image]: Row pitch must be >= height * host row pitch.");

	// map image region
	cl_event write_event;
	std::size_t row_pitch{0ull};
	std::size_t slice_pitch{0ull};
	// cast mapped pointer to uint8_t. This way we are allowed to do byte-wise pointer arithmetic.
	CL_EX(clEnqueueWriteImage(
		m_cl_state->command_queue(),
		m_image,
		blocking ? CL_TRUE : CL_FALSE,
		&origin[0],
		&region[0],
		host_row_pitch,
		host_slice_pitch,
		data_ptr,
		static_cast<cl_uint>(m_event_cache.size()),
		(m_event_cache.size() > 0ull ? m_event_cache.data() : nullptr),
		&write_event
	));	

	// unmap image and return event
	return Event{write_event};
}

simple_cl::cl::Event simple_cl::cl::Image::img_read_mapped(const ImageRegion& img_region, const HostFormat& format, void* data_ptr, ChannelDefaultValue default_value)
{
	if(m_image_desc.flags.host_access == HostAccess::NoAccess || m_image_desc.flags.host_access == HostAccess::WriteOnly)
		throw std::runtime_error("[Image]: Host is not allowed to read this image.");
	if(!(img_region.dimensions.width && img_region.dimensions.height && img_region.dimensions.depth))
		throw std::runtime_error("[Image]: Read failed, region is empty.");
	// check if region matches
	if((img_region.offset.offset_width + img_region.dimensions.width > m_image_desc.dimensions.width) ||
		(img_region.offset.offset_height + img_region.dimensions.height > m_image_desc.dimensions.height) ||
		(img_region.offset.offset_depth + img_region.dimensions.depth > m_image_desc.dimensions.depth))
		throw std::runtime_error("[Image]: Read failed. Input region exceeds image dimensions.");
	// handle wrong pitch values
	if((m_image_desc.type == ImageType::Image1D || m_image_desc.type == ImageType::Image2D) && format.pitch.slice_pitch != 0ull)
		throw std::runtime_error("[Image]: Slice pitch must be 0 for 1D or 2D images.");

	// for parameterization of clEnqueueMapImage
	std::size_t origin[]{img_region.offset.offset_width, img_region.offset.offset_height, img_region.offset.offset_depth};
	std::size_t region[]{img_region.dimensions.width, img_region.dimensions.height, img_region.dimensions.depth};

	// pixel sizes for cl and host
	std::size_t cl_component_size = get_image_channel_type_size(m_image_desc.channel_type);
	std::size_t host_component_size = get_host_channel_type_size(format.channel_type);
	std::size_t cl_num_components = get_num_image_pixel_components(m_image_desc.channel_order);
	std::size_t host_num_components = get_num_host_pixel_components(format.channel_order);
	std::size_t cl_pixel_size = cl_component_size * cl_num_components;
	std::size_t host_pixel_size = host_component_size * host_num_components;

	// pitches for host in bytes
	std::size_t host_row_pitch = (format.pitch.row_pitch != 0ull ? format.pitch.row_pitch : img_region.dimensions.width * host_pixel_size);
	if(host_row_pitch < img_region.dimensions.width * host_pixel_size)
		throw std::runtime_error("[Image]: Row pitch must be >= region width * bytes per pixel.");
	std::size_t host_slice_pitch = (format.pitch.slice_pitch != 0ull ? format.pitch.slice_pitch : img_region.dimensions.height * host_row_pitch);
	if(host_slice_pitch < img_region.dimensions.height * host_row_pitch)
		throw std::runtime_error("[Image]: Row pitch must be >= height * host row pitch.");

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
		throw CLException(err, __LINE__, __FILE__, "[Image]: clEnqueueMapImage failed.");

	// if slice_pitch is 0 we have a 1D o 2D image. Re-use slice_pitch in this case:
	slice_pitch = slice_pitch ? slice_pitch : row_pitch * img_region.dimensions.height;
	// determine size of copied memory regions
	std::size_t row_size = std::min(row_pitch, host_row_pitch);
	std::size_t slice_size = std::min(slice_pitch, host_slice_pitch);
	std::size_t region_size = img_region.dimensions.depth * host_slice_pitch;

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
				for(std::size_t slice_idx = 0; slice_idx < img_region.dimensions.depth; ++slice_idx) // copy one slice at a time
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
				for(std::size_t slice_idx = 0; slice_idx < img_region.dimensions.depth; ++slice_idx)
				{
					const uint8_t* cur_row_img_ptr = cur_img_ptr;
					uint8_t* cur_row_data_ptr = cur_data_ptr;
					for(std::size_t row_idx = 0; row_idx < img_region.dimensions.height; ++row_idx) // copy row by row
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
		throw std::runtime_error("[Image]: Image read failed. Host format does not match image format.");

	// unmap image and return event
	CL_EX(clEnqueueUnmapMemObject(m_cl_state->command_queue(), m_image, img_ptr, 0ull, nullptr, &map_event));
	return Event{map_event};
}

simple_cl::cl::Event simple_cl::cl::Image::img_read(const ImageRegion& img_region, const HostFormat& format, void* data_ptr, bool blocking, ChannelDefaultValue default_value)
{
	if(m_image_desc.flags.host_access == HostAccess::NoAccess || m_image_desc.flags.host_access == HostAccess::WriteOnly)
		throw std::runtime_error("[Image]: Host is not allowed to read this image.");
	if(!(img_region.dimensions.width && img_region.dimensions.height && img_region.dimensions.depth))
		throw std::runtime_error("[Image]: Read failed, region is empty.");
	// check if region matches
	if((img_region.offset.offset_width + img_region.dimensions.width > m_image_desc.dimensions.width) ||
		(img_region.offset.offset_height + img_region.dimensions.height > m_image_desc.dimensions.height) ||
		(img_region.offset.offset_depth + img_region.dimensions.depth > m_image_desc.dimensions.depth))
		throw std::runtime_error("[Image]: Read failed. Input region exceeds image dimensions.");
	// handle wrong pitch values
	if((m_image_desc.type == ImageType::Image1D || m_image_desc.type == ImageType::Image2D) && format.pitch.slice_pitch != 0ull)
		throw std::runtime_error("[Image]: Slice pitch must be 0 for 1D or 2D images.");

	// ensure matching image format
	if(!match_format(format))
		throw std::runtime_error("[Image]: Read failed. Host format doesn't match image format.");

	// for parameterization of clEnqueueMapImage
	std::size_t origin[]{img_region.offset.offset_width, img_region.offset.offset_height, img_region.offset.offset_depth};
	std::size_t region[]{img_region.dimensions.width, img_region.dimensions.height, img_region.dimensions.depth};

	// pixel sizes for cl and host
	std::size_t cl_component_size = get_image_channel_type_size(m_image_desc.channel_type);
	std::size_t host_component_size = get_host_channel_type_size(format.channel_type);
	std::size_t cl_num_components = get_num_image_pixel_components(m_image_desc.channel_order);
	std::size_t host_num_components = get_num_host_pixel_components(format.channel_order);
	std::size_t cl_pixel_size = cl_component_size * cl_num_components;
	std::size_t host_pixel_size = host_component_size * host_num_components;

	// pitches for host in bytes
	std::size_t host_row_pitch = (format.pitch.row_pitch != 0ull ? format.pitch.row_pitch : img_region.dimensions.width * host_pixel_size);
	if(host_row_pitch < img_region.dimensions.width * host_pixel_size)
		throw std::runtime_error("[Image]: Row pitch must be >= region width * bytes per pixel.");
	std::size_t host_slice_pitch = (format.pitch.slice_pitch != 0ull ? format.pitch.slice_pitch : img_region.dimensions.height * host_row_pitch);
	if(host_slice_pitch < img_region.dimensions.height * host_row_pitch)
		throw std::runtime_error("[Image]: Row pitch must be >= height * host row pitch.");

	// map image region
	cl_event read_event;
	std::size_t row_pitch{0ull};
	std::size_t slice_pitch{0ull};
	// cast mapped pointer to uint8_t. This way we are allowed to do byte-wise pointer arithmetic.
	CL_EX(clEnqueueReadImage(
		m_cl_state->command_queue(),
		m_image,
		blocking ? CL_TRUE : CL_FALSE,
		&origin[0],
		&region[0],
		host_row_pitch,
		host_slice_pitch,
		data_ptr,
		static_cast<cl_uint>(m_event_cache.size()),
		(m_event_cache.size() > 0ull ? m_event_cache.data() : nullptr),
		&read_event
	));

	return Event{read_event};
}

simple_cl::cl::Event simple_cl::cl::Image::img_fill(const FillColor& color, const ImageRegion& img_region)
{
	if(m_image_desc.flags.host_access == HostAccess::NoAccess || m_image_desc.flags.host_access == HostAccess::ReadOnly)
		throw std::runtime_error("[Image]: Host is not allowed to fill this image.");
	if(!(img_region.dimensions.width && img_region.dimensions.height && img_region.dimensions.depth))
		throw std::runtime_error("[Image]: Fill failed, region is empty.");
	if((img_region.offset.offset_width + img_region.dimensions.width > m_image_desc.dimensions.width) ||
		(img_region.offset.offset_height + img_region.dimensions.height > m_image_desc.dimensions.height) ||
		(img_region.offset.offset_depth + img_region.dimensions.depth > m_image_desc.dimensions.depth))
		throw std::runtime_error("[Image]: Fill failed. Input region exceeds image dimensions.");

	// --- prepare color data
	// largest possible fill color: 4x4 bytes
	alignas(meta::max_align_of<float, uint32_t, int32_t>::value) uint8_t color_buffer[meta::max_size_of<float, uint32_t, int32_t>::value * 4ull];
	// Three cases: float, int, normalized int
	ChannelBaseType base_type{get_image_channel_base_type(m_image_desc.channel_type)};
	std::size_t	type_size{get_image_channel_type_size(m_image_desc.channel_type)};
	bool normalized_int{is_image_channel_format_normalized_integer(m_image_desc.channel_type)};

	std::size_t channel_indices[]{
		std::size_t(static_cast<uint8_t>(get_image_color_channel(m_image_desc.channel_order, 0ull))),
		std::size_t(static_cast<uint8_t>(get_image_color_channel(m_image_desc.channel_order, 1ull))),
		std::size_t(static_cast<uint8_t>(get_image_color_channel(m_image_desc.channel_order, 2ull))),
		std::size_t(static_cast<uint8_t>(get_image_color_channel(m_image_desc.channel_order, 3ull))),
	};
	
	if(base_type == ChannelBaseType::Float || normalized_int) // In case of floating point format or normalized integer format, create array of floats.
	{
		static_cast<float*>(static_cast<void*>(&color_buffer[0]))[0] = color.get(channel_indices[0]);
		static_cast<float*>(static_cast<void*>(&color_buffer[0]))[1] = color.get(channel_indices[1]);
		static_cast<float*>(static_cast<void*>(&color_buffer[0]))[2] = color.get(channel_indices[2]);
		static_cast<float*>(static_cast<void*>(&color_buffer[0]))[3] = color.get(channel_indices[3]);
	}
	else if(base_type == ChannelBaseType::Int) // Signed integers
	{
		switch(type_size)
		{
			case 1ull:
				static_cast<int8_t*>(static_cast<void*>(&color_buffer[0]))[0] = static_cast<int8_t>(color.get(channel_indices[0]));
				static_cast<int8_t*>(static_cast<void*>(&color_buffer[0]))[1] = static_cast<int8_t>(color.get(channel_indices[1]));
				static_cast<int8_t*>(static_cast<void*>(&color_buffer[0]))[2] = static_cast<int8_t>(color.get(channel_indices[2]));
				static_cast<int8_t*>(static_cast<void*>(&color_buffer[0]))[3] = static_cast<int8_t>(color.get(channel_indices[3]));
				break;
			case 2ull:
				static_cast<int16_t*>(static_cast<void*>(&color_buffer[0]))[0] = static_cast<int16_t>(color.get(channel_indices[0]));
				static_cast<int16_t*>(static_cast<void*>(&color_buffer[0]))[1] = static_cast<int16_t>(color.get(channel_indices[1]));
				static_cast<int16_t*>(static_cast<void*>(&color_buffer[0]))[2] = static_cast<int16_t>(color.get(channel_indices[2]));
				static_cast<int16_t*>(static_cast<void*>(&color_buffer[0]))[3] = static_cast<int16_t>(color.get(channel_indices[3]));
				break;
			case 4ull:
				static_cast<int32_t*>(static_cast<void*>(&color_buffer[0]))[0] = static_cast<int32_t>(color.get(channel_indices[0]));
				static_cast<int32_t*>(static_cast<void*>(&color_buffer[0]))[1] = static_cast<int32_t>(color.get(channel_indices[1]));
				static_cast<int32_t*>(static_cast<void*>(&color_buffer[0]))[2] = static_cast<int32_t>(color.get(channel_indices[2]));
				static_cast<int32_t*>(static_cast<void*>(&color_buffer[0]))[3] = static_cast<int32_t>(color.get(channel_indices[3]));
				break;
			default:
				throw std::runtime_error("[Image]: Fill failed. Invalid channel type size");
				break;
		}
	}
	else // Unsigned integers
	{
		switch(type_size)
		{
			case 1ull:
				static_cast<uint8_t*>(static_cast<void*>(&color_buffer[0]))[0] = static_cast<uint8_t>(color.get(channel_indices[0]));
				static_cast<uint8_t*>(static_cast<void*>(&color_buffer[0]))[1] = static_cast<uint8_t>(color.get(channel_indices[1]));
				static_cast<uint8_t*>(static_cast<void*>(&color_buffer[0]))[2] = static_cast<uint8_t>(color.get(channel_indices[2]));
				static_cast<uint8_t*>(static_cast<void*>(&color_buffer[0]))[3] = static_cast<uint8_t>(color.get(channel_indices[3]));
				break;
			case 2ull:
				static_cast<uint16_t*>(static_cast<void*>(&color_buffer[0]))[0] = static_cast<uint16_t>(color.get(channel_indices[0]));
				static_cast<uint16_t*>(static_cast<void*>(&color_buffer[0]))[1] = static_cast<uint16_t>(color.get(channel_indices[1]));
				static_cast<uint16_t*>(static_cast<void*>(&color_buffer[0]))[2] = static_cast<uint16_t>(color.get(channel_indices[2]));
				static_cast<uint16_t*>(static_cast<void*>(&color_buffer[0]))[3] = static_cast<uint16_t>(color.get(channel_indices[3]));
				break;
			case 4ull:
				static_cast<uint32_t*>(static_cast<void*>(&color_buffer[0]))[0] = static_cast<uint32_t>(color.get(channel_indices[0]));
				static_cast<uint32_t*>(static_cast<void*>(&color_buffer[0]))[1] = static_cast<uint32_t>(color.get(channel_indices[1]));
				static_cast<uint32_t*>(static_cast<void*>(&color_buffer[0]))[2] = static_cast<uint32_t>(color.get(channel_indices[2]));
				static_cast<uint32_t*>(static_cast<void*>(&color_buffer[0]))[3] = static_cast<uint32_t>(color.get(channel_indices[3]));
				break;
			default:
				throw std::runtime_error("[Image]: Fill failed. Invalid channel type size");
				break;
		}
	}
	
	// image region
	std::size_t origin[]{img_region.offset.offset_width, img_region.offset.offset_height, img_region.offset.offset_depth};
	std::size_t region[]{img_region.dimensions.width, img_region.dimensions.height, img_region.dimensions.depth};

	// API call
	cl_event fill_event{nullptr};
	CL_EX(clEnqueueFillImage(
		m_cl_state->command_queue(),
		m_image,
		static_cast<const void*>(&color_buffer[0]),
		&origin[0],
		&region[0],
		static_cast<cl_uint>(m_event_cache.size()),
		(m_event_cache.size() > 0ull ? m_event_cache.data() : nullptr),
		&fill_event)
	);
	return Event{fill_event};
}

#pragma endregion
#pragma endregion
