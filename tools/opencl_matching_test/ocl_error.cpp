#include <ocl_error.hpp>
#include <iostream>

const char* ocl_template_matching::_get_cl_error_string(cl_int error_val)
{
	const char* err = nullptr;
	switch(error_val)
	{
		case CL_SUCCESS: err = "CL_SUCCESS"; break;
		case CL_DEVICE_NOT_FOUND: err = "CL_DEVICE_NOT_FOUND"; break;
		case CL_DEVICE_NOT_AVAILABLE: err = "CL_DEVICE_NOT_AVAILABLE"; break;
		case CL_COMPILER_NOT_AVAILABLE: err = "CL_COMPILER_NOT_AVAILABLE"; break;
		case CL_MEM_OBJECT_ALLOCATION_FAILURE: err = "CL_MEM_OBJECT_ALLOCATION_FAILURE"; break;
		case CL_OUT_OF_RESOURCES: err = "CL_OUT_OF_RESOURCES"; break;
		case CL_OUT_OF_HOST_MEMORY: err = "CL_OUT_OF_HOST_MEMORY"; break;
		case CL_PROFILING_INFO_NOT_AVAILABLE: err = "CL_PROFILING_INFO_NOT_AVAILABLE"; break;
		case CL_MEM_COPY_OVERLAP: err = "CL_MEM_COPY_OVERLAP"; break;
		case CL_IMAGE_FORMAT_MISMATCH: err = "CL_IMAGE_FORMAT_MISMATCH"; break;
		case CL_IMAGE_FORMAT_NOT_SUPPORTED: err = "CL_IMAGE_FORMAT_NOT_SUPPORTED"; break;
		case CL_BUILD_PROGRAM_FAILURE: err = "CL_BUILD_PROGRAM_FAILURE"; break;
		case CL_MAP_FAILURE: err = "CL_MAP_FAILURE"; break;
		case CL_MISALIGNED_SUB_BUFFER_OFFSET: err = "CL_MISALIGNED_SUB_BUFFER_OFFSET"; break;
		case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: err = "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST"; break;
		case CL_COMPILE_PROGRAM_FAILURE: err = "CL_COMPILE_PROGRAM_FAILURE"; break;
		case CL_LINKER_NOT_AVAILABLE: err = "CL_LINKER_NOT_AVAILABLE"; break;
		case CL_LINK_PROGRAM_FAILURE: err = "CL_LINK_PROGRAM_FAILURE"; break;
		case CL_DEVICE_PARTITION_FAILED: err = "CL_DEVICE_PARTITION_FAILED"; break;
		case CL_KERNEL_ARG_INFO_NOT_AVAILABLE: err = "CL_KERNEL_ARG_INFO_NOT_AVAILABLE"; break;
		case CL_INVALID_VALUE: err = "CL_INVALID_VALUE"; break;
		case CL_INVALID_DEVICE_TYPE: err = "CL_INVALID_DEVICE_TYPE"; break;
		case CL_INVALID_PLATFORM: err = "CL_INVALID_PLATFORM"; break;
		case CL_INVALID_DEVICE: err = "CL_INVALID_DEVICE"; break;
		case CL_INVALID_CONTEXT: err = "CL_INVALID_CONTEXT"; break;
		case CL_INVALID_QUEUE_PROPERTIES: err = "CL_INVALID_QUEUE_PROPERTIES"; break;
		case CL_INVALID_COMMAND_QUEUE: err = "CL_INVALID_COMMAND_QUEUE"; break;
		case CL_INVALID_HOST_PTR: err = "CL_INVALID_HOST_PTR"; break;
		case CL_INVALID_MEM_OBJECT: err = "CL_INVALID_MEM_OBJECT"; break;
		case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: err = "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR"; break;
		case CL_INVALID_IMAGE_SIZE: err = "CL_INVALID_IMAGE_SIZE"; break;
		case CL_INVALID_SAMPLER: err = "CL_INVALID_SAMPLER"; break;
		case CL_INVALID_BINARY: err = "CL_INVALID_BINARY"; break;
		case CL_INVALID_BUILD_OPTIONS: err = "CL_INVALID_BUILD_OPTIONS"; break;
		case CL_INVALID_PROGRAM: err = "CL_INVALID_PROGRAM"; break;
		case CL_INVALID_PROGRAM_EXECUTABLE: err = "CL_INVALID_PROGRAM_EXECUTABLE"; break;
		case CL_INVALID_KERNEL_NAME: err = "CL_INVALID_KERNEL_NAME"; break;
		case CL_INVALID_KERNEL_DEFINITION: err = "CL_INVALID_KERNEL_DEFINITION"; break;
		case CL_INVALID_KERNEL: err = "CL_INVALID_KERNEL"; break;
		case CL_INVALID_ARG_INDEX: err = "CL_INVALID_ARG_INDEX"; break;
		case CL_INVALID_ARG_VALUE: err = "CL_INVALID_ARG_VALUE"; break;
		case CL_INVALID_ARG_SIZE: err = "CL_INVALID_ARG_SIZE"; break;
		case CL_INVALID_KERNEL_ARGS: err = "CL_INVALID_KERNEL_ARGS"; break;
		case CL_INVALID_WORK_DIMENSION: err = "CL_INVALID_WORK_DIMENSION"; break;
		case CL_INVALID_WORK_GROUP_SIZE: err = "CL_INVALID_WORK_GROUP_SIZE"; break;
		case CL_INVALID_WORK_ITEM_SIZE: err = "CL_INVALID_WORK_ITEM_SIZE"; break;
		case CL_INVALID_GLOBAL_OFFSET: err = "CL_INVALID_GLOBAL_OFFSET"; break;
		case CL_INVALID_EVENT_WAIT_LIST: err = "CL_INVALID_EVENT_WAIT_LIST"; break;
		case CL_INVALID_EVENT: err = "CL_INVALID_EVENT"; break;
		case CL_INVALID_OPERATION: err = "CL_INVALID_OPERATION"; break;
		case CL_INVALID_GL_OBJECT: err = "CL_INVALID_GL_OBJECT"; break;
		case CL_INVALID_BUFFER_SIZE: err = "CL_INVALID_BUFFER_SIZE"; break;
		case CL_INVALID_MIP_LEVEL: err = "CL_INVALID_MIP_LEVEL"; break;
		case CL_INVALID_GLOBAL_WORK_SIZE: err = "CL_INVALID_GLOBAL_WORK_SIZE"; break;
		case CL_INVALID_PROPERTY: err = "CL_INVALID_PROPERTY"; break;
		case CL_INVALID_IMAGE_DESCRIPTOR: err = "CL_INVALID_IMAGE_DESCRIPTOR"; break;
		case CL_INVALID_COMPILER_OPTIONS: err = "CL_INVALID_COMPILER_OPTIONS"; break;
		case CL_INVALID_LINKER_OPTIONS: err = "CL_INVALID_LINKER_OPTIONS"; break;
		case CL_INVALID_DEVICE_PARTITION_COUNT: err = "CL_INVALID_DEVICE_PARTITION_COUNT"; break;
		case CL_INVALID_PIPE_SIZE: err = "CL_INVALID_PIPE_SIZE"; break;
		case CL_INVALID_DEVICE_QUEUE: err = "CL_INVALID_DEVICE_QUEUE"; break;
		default: err = "UNKNOWN_ERROR"; break;
	}
	return err;
}

// print error if there is any
cl_int ocl_template_matching::_print_cl_error(cl_int error_val, const char* file, int line)
{
	if(error_val == CL_SUCCESS)
		return error_val;
	std::cerr << "[OpenCL ERROR]: (File: \"" << file << "\", Line: " << line << "):" << std::endl;
	std::cerr << _get_cl_error_string(error_val) << std::endl;
	return error_val;
}

ocl_template_matching::CLException::CLException() :
	cl_error_val{0},
	line{0},
	file{""}
{
}

ocl_template_matching::CLException::CLException(cl_int error, int _line, const char* _file) :
	cl_error_val{error},
	line{_line},
	file{_file}
{
}

const char* ocl_template_matching::CLException::what() const noexcept
{
	return _get_cl_error_string(cl_error_val);
}

// throw if there is a cl error
cl_int ocl_template_matching::_check_throw_cl_error(cl_int error_val, const char* file, int line)
{
	if(error_val != CL_SUCCESS)
		throw CLException{error_val, line, file};
	return error_val;
}