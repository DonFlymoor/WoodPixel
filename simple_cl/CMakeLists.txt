cmake_minimum_required(VERSION 3.1)

set(lib_name simple_cl)

project(${lib_name} LANGUAGES CXX)

# depends on OpenCL but only the basic cl.h header. No fancy bindings needed.
find_package(OpenCL REQUIRED) # creates target OpenCL::OpenCL

add_library(${lib_name} STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/simple_cl.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/simple_cl_error.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/simple_cl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/simple_cl_error.cpp
)

target_link_libraries(${lib_name}
    PUBLIC
        OpenCL::OpenCL
)

target_compile_definitions(${lib_name}
	PUBLIC
		$<$<CONFIG:Debug>:CLERR_DEBUG>
        CL_TARGET_OPENCL_VERSION=120 # restrict outselves to OpenCL 1.2 as it is available virtually everywhere
)

target_compile_features(${lib_name} PUBLIC cxx_std_11)

target_include_directories(${lib_name}
    INTERFACE 
        $<INSTALL_INTERFACE:include>    
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(${lib_name}
    PROPERTIES
        DEBUG_POSTFIX "d"
)

install(TARGETS ${lib_name}
        EXPORT "${lib_name}Config"
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
)

install(DIRECTORY include DESTINATION include)

install(EXPORT "${lib_name}Config" DESTINATION lib/cmake/${lib_name})