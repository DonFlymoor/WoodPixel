/** \file simple_cl_error.h
*	\author Fabian Friederichs
*
*	\brief Customized exception class and error handling macros.
*/

#ifndef _SIMPLE_CL_ERROR_HPP_
#define _SIMPLE_CL_ERROR_HPP_

#include <CL/cl.h>
#include <exception>
#include <stdexcept>

namespace simple_cl
{
	// generate human readable error string
	const char* _get_cl_error_string(cl_int error_val);
	// print error if there is any
	cl_int _print_cl_error(cl_int error_val, const char* file, int line);	
	// throw if there is a cl error
	cl_int _check_throw_cl_error(cl_int error_val, const char* file, int line);

	// exception class
	class CLException : public std::exception
	{
	public:
		CLException();
		CLException(cl_int error, int _line = 0, const char* _file = nullptr, const char* errormsg = nullptr);

		CLException(const CLException&) noexcept = default;
		CLException(CLException&&) noexcept = default;
		CLException& operator=(const CLException&) noexcept = default;
		CLException& operator=(CLException&&) noexcept = default;

		virtual const char* what() const noexcept override;
	private:
		const cl_int cl_error_val;
		int line;
		const char* file;
		const char* additional_info;
	};	
}

#ifdef CLERR_DEBUG
	#define CL(clcall) simple_cl::_print_cl_error(clcall, __FILE__, __LINE__) 
#else
	#define CL(clcall) clcall
#endif

#define CL_EX(clcall) simple_cl::_check_throw_cl_error(clcall, __FILE__, __LINE__)
#endif