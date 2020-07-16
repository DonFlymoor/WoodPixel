CMAKE_MINIMUM_REQUIRED(VERSION 3.1)

SET(LIB_NAME trlib)

SET(SRC
	adaptive_patch.hpp
	affine_transformation.cpp
	affine_transformation.hpp
	bezier_curve.cpp
	bezier_curve.hpp
	bezier_curve_impl.hpp
	bezier_ransac.cpp
	bezier_ransac.hpp
	blob.cpp
	blob.hpp
	convex_hull.cpp
	convex_hull.hpp
	cumulative_distribution_function.cpp
	cumulative_distribution_function.hpp
	cut_saver.cpp
	cut_saver.hpp
	cut_saver_impl.hpp
	data_grid.hpp
	draw.cpp
	draw.hpp
	eps_saver.cpp
	eps_saver.hpp
	feature_detect.cpp
	feature_detect.hpp
	feature_evaluator.cpp
	feature_evaluator.hpp
	feature_vector.cpp
	feature_vector.hpp
	flood_fill.hpp
	gabor_filter.cpp
	gabor_filter.hpp
	gabor_filter_bank.cpp
	gabor_filter_bank.hpp
	generate_patches.cpp
	generate_patches.hpp
	gh_invariant.hpp
	grid.cpp
	grid.hpp
	hershey_font.cpp
	hershey_font.hpp
	histogram.cpp
	histogram.hpp
	histogram_vector.hpp
	hough_circle_detect.cpp
	hough_circle_detect.hpp
	line.cpp
	line.hpp
	line_segment_detect.cpp
	line_segment_detect.hpp
	linspace.hpp
	mat.hpp
	material_panel.cpp
	material_panel.hpp
	math.hpp
	merge_patch.cpp
	merge_patch.hpp
	opencv_extra.hpp
	patch.cpp
	patch.hpp
	patch_region.cpp
	patch_region.hpp
	patch_region_impl.hpp
	print.hpp
	print_debug.cpp
	print_debug.hpp
	rectangle_patch.hpp
	serializable.hpp
	sort_pca.hpp
	svg_saver.cpp
	svg_saver.hpp
	texture.cpp
	texture.hpp
	texture_marker.cpp
	texture_marker.hpp
	timer.hpp
	transformations.cpp
	transformations.hpp
	tree_match.cpp
	tree_match.hpp
	tree_match_gpu.hpp
	tree_match_gpu.cpp
	vector.hpp
	vector_graphics_saver.cpp
	vector_graphics_saver.hpp
	idgen.hpp
	${PROJECT_SOURCE_DIR}/config.h
)

SET(trlib_enable_opencl_template_matching OFF CACHE BOOL "Enables or disables opencl patch matching features.")
if(${trlib_enable_opencl_template_matching})
	list(APPEND SRC
		ocl_patch_matcher.hpp
		ocl_patch_matcher.cpp
		matching_policies.hpp
		matching_policies.cpp
	)
	# opencl kernels
	include(include_cl_kernel)
	include_cl_kernel("kernels/sqdiff_naive.cl" "kernels/cl_kernel.hpp.in" "kernels/kernel_sqdiff_naive.hpp" "-cl-std=CL1.2")
	include_cl_kernel("kernels/sqdiff_constant_kernel.cl" "kernels/cl_kernel.hpp.in" "kernels/kernel_sqdiff_constant_kernel.hpp" "-cl-std=CL1.2")
	include_cl_kernel("kernels/erode.cl" "kernels/cl_kernel.hpp.in" "kernels/kernel_erode.hpp" "-cl-std=CL1.2")
	include_cl_kernel("kernels/find_min.cl" "kernels/cl_kernel.hpp.in" "kernels/find_min.hpp" "-cl-std=CL1.2")
endif()

ADD_LIBRARY(${LIB_NAME} ${SRC})
      
TARGET_INCLUDE_DIRECTORIES(${LIB_NAME} PUBLIC "${CMAKE_CURRENT_LIST_DIR}")
TARGET_LINK_LIBRARIES(${LIB_NAME} PUBLIC LIBS_ALLDEPS)

if(${trlib_enable_opencl_template_matching})
	TARGET_LINK_LIBRARIES(${LIB_NAME} PUBLIC simple_cl)
	TARGET_INCLUDE_DIRECTORIES(${LIB_NAME} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
	TARGET_COMPILE_DEFINITIONS(${LIB_NAME} PUBLIC TRLIB_TREE_MATCH_USE_OPENCL)
endif()