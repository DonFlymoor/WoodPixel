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
	/// generate human readable error string
	const char* _get_cl_error_string(cl_int error_val);
	/// print error if there is any
	cl_int _print_cl_error(cl_int error_val, const char* file, int line);	
	/// throw if there is a cl error
	cl_int _check_throw_cl_error(cl_int error_val, const char* file, int line);

	/**
	 *	\brief	Custom exception class for OpenCL related errors. 
	*/
	class CLException : public std::exception
	{
	public:
		/// Default constructor.
		CLException();
		/**
		 *	\brief			Creates new CLException instance.
		 *	\param error	OpenCL error code.
		 *	\param _line	Line number which shall be reported in the error message.
		 *	\param _file	File name which shall be reported in the error message.
		 *	\param errormsg Additional error message string. Is appended at the end of the error message.
		 *	\return 
		*/
		CLException(cl_int error, int _line = 0, const char* _file = nullptr, const char* errormsg = nullptr);

		/// Copy constructor.
		CLException(const CLException&) noexcept = default;
		/// Move constructor.
		CLException(CLException&&) noexcept = default;
		/// Copy assignment.
		CLException& operator=(const CLException&) noexcept = default;
		/// Move assignment.
		CLException& operator=(CLException&&) noexcept = default;

		/**
		 *	\brief	Returns the error message for this exception.
		 *	\return Error message.
		*/
		virtual const char* what() const noexcept override;
	private:
		const cl_int cl_error_val;		///<	OpenCL error code.
		int line;						///<	Line this exception was raised from.
		const char* file;				///<	File this exception was raised from.
		const char* additional_info;	///<	Additional info string.
	};	
}

#ifdef CLERR_DEBUG
	/// If in debug build, reports error code and line + file if the enclosed OpenCL call does not return CL_SUCCESS.
	#define CL(clcall) simple_cl::_print_cl_error(clcall, __FILE__, __LINE__) 
#else
	/// In release build, this macro does nothing.
	#define CL(clcall) clcall
#endif

/// Throws a CLException if the enclosed OpenCL call does not return CL_SUCCESS.
#define CL_EX(clcall) simple_cl::_check_throw_cl_error(clcall, __FILE__, __LINE__)
#endif