CMAKE_MINIMUM_REQUIRED(VERSION 3.1)

SET(EXECUTABLE_NAME opencl_matching_test)

SET(SRC
	opencl_matching_test.cpp
	ocl_template_matcher.hpp
	ocl_template_matcher.cpp
	matching_policies.hpp
	matching_policies.cpp
)

ADD_EXECUTABLE(${EXECUTABLE_NAME} ${SRC})

TARGET_LINK_LIBRARIES(${EXECUTABLE_NAME} simple_cl LIBS_OPENCV trlib)

TARGET_INCLUDE_DIRECTORIES(${EXECUTABLE_NAME}
	PUBLIC
		${CMAKE_CURRENT_SOURCE_DIR}
)