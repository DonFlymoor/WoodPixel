# includes kernel source files and puts them into a dedicated header.
function(include_cl_kernel kernel_file template_header destination_header kernel_compiler_options)
    # read kernel source
    file(READ "${kernel_file}" kernel_source)
    # get kernel name
    get_filename_component(kernel_name "${kernel_file}" NAME_WE)
    # generate header which includes the kernel source
    configure_file("${template_header}" "${destination_header}" NEWLINE_STYLE LF)
endfunction()

### Example for the template header. File should have ending *.hpp.in or *.h.in if we follow common practice.
# #ifndef _CL_KERNEL_${kernel_name}_HPP_
# #define _CL_KERNEL_${kernel_name}_HPP_
#
# namespace kernels
# {
# 	const char* ${kernel_name}_src = R"(${kernel_source})";
#   const char* ${kernel_name}_copt = "${kernel_compiler_options}";
# }
#
# #endif