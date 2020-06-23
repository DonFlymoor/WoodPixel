CMAKE_MINIMUM_REQUIRED(VERSION 3.1)

SET(EXECUTABLE_NAME opencl_matching_test)

SET(SRC
	opencl_matching_test.cpp
	ocl_template_matcher.hpp
	ocl_template_matcher.cpp
	ocl_error.hpp
	ocl_error.cpp
	ocl_wrappers.hpp
	ocl_wrappers.cpp
	matching_policies.hpp
	# ${PROJECT_SOURCE_DIR}/config.h
)

ADD_EXECUTABLE(${EXECUTABLE_NAME} ${SRC})

TARGET_LINK_LIBRARIES(${EXECUTABLE_NAME} LIBS_OPENCL LIBS_OPENCV LIBS_TRLIB)

TARGET_COMPILE_DEFINITIONS(${EXECUTABLE_NAME}
	PUBLIC
		$<$<CONFIG:Debug>:CLERR_DEBUG>
		CL_USE_DEPRECATED_OPENCL_1_2_APIS) # when we finally put the matching stuff into the trlib project, let's see if clients have to know opencl stuff then.

TARGET_INCLUDE_DIRECTORIES(${EXECUTABLE_NAME}
	PUBLIC
		${CMAKE_CURRENT_SOURCE_DIR}
)