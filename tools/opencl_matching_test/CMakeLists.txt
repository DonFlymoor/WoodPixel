CMAKE_MINIMUM_REQUIRED(VERSION 3.1)

include(include_cl_kernel)

SET(EXECUTABLE_NAME opencl_matching_test)

SET(SRC
	opencl_matching_test.cpp
	ocl_template_matcher.hpp
	ocl_template_matcher.cpp
	matching_policies.hpp
	matching_policies.cpp
)

# opencl kernels
include_cl_kernel("kernels/sqdiff_naive.cl" "kernels/cl_kernel.hpp.in" "kernels/kernel_sqdiff_naive.hpp" "-cl-std=CL1.2")
include_cl_kernel("kernels/sqdiff_constant_kernel.cl" "kernels/cl_kernel.hpp.in" "kernels/kernel_sqdiff_constant_kernel.hpp" "-cl-std=CL1.2")

ADD_EXECUTABLE(${EXECUTABLE_NAME} ${SRC})

TARGET_LINK_LIBRARIES(${EXECUTABLE_NAME} simple_cl LIBS_OPENCV trlib)

TARGET_INCLUDE_DIRECTORIES(${EXECUTABLE_NAME}
	PUBLIC
		${CMAKE_CURRENT_SOURCE_DIR}
	PRIVATE
		${CMAKE_CURRENT_BINARY_DIR}
)