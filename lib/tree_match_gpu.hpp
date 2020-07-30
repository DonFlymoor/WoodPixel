/*
WoodPixel - Supplementary code for Computational Parquetry:
			Fabricated Style Transfer with Wood Pixels
			ACM Transactions on Graphics 39(2), 2020

Copyright (C) 2020  Julian Iseringhausen, University of Bonn, <iseringhausen@cs.uni-bonn.de>
Copyright (C) 2020  Matthias Hullin, University of Bonn, <hullin@cs.uni-bonn.de>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#ifndef TRLIB_TREE_MATCH_GPU_HPP_
#define TRLIB_TREE_MATCH_GPU_HPP_

#include <deque>
#include <vector>

#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#ifdef TRLIB_TREE_MATCH_USE_OPENCL
#include <ocl_patch_matcher.hpp>
#include <matching_policies.hpp>
#endif

#include "adaptive_patch.hpp"
#include "feature_evaluator.hpp"
#include "gabor_filter_bank.hpp"
#include "grid.hpp"
#include "patch.hpp"
#include "texture.hpp"

/**
 *	\brief	Variant of the TreeMatch class which provides code paths which uses the new OpenCL matching classes for accelerating the matching process.
 *	If TRLIB_TREE_MATCH_USE_OPENCL is defined, the OpenCL codepaths are enabled, otherwise it behaves exactly like the old matcher class.
 */
class TreeMatchGPU
{
public:
#ifdef TRLIB_TREE_MATCH_USE_OPENCL
	/**
	 *	\brief	Packs options for the OpenCL template matching code path.
	*/
	struct GPUMatchingOptions
	{
		ocl_patch_matching::Matcher::DeviceSelectionPolicy device_selection_policy = ocl_patch_matching::Matcher::DeviceSelectionPolicy::MostComputeUnits; ///< Specifies how to choose the GPU device if there are more than one.
		std::size_t max_texture_cache_memory = 536870912ull;	///< Maximum GPU memory to use for caching input textures. Currently ignored.
		std::size_t max_num_kernel_pixels_gpu = 64ull * 64ull;	///< Maximum number of pixels in a kernel for which the OpenCL matching variant is applied.
		std::size_t local_block_size = 16ull;					///< Local work group size (total work group size in number of processing elements is this quantity squared!).
		std::size_t constant_kernel_max_pixels = 500 * 500ull;	///< Maximum number of kernel pixels for which the constant buffer optimization shall be used.
		std::size_t max_local_pixels = 4096ull;					///< Maximum number of image window pixels for which the local (shared) memory optimization shall be used.
		std::size_t max_rotations_per_pass = 1ull;				///< Batch size for the processing of input texture rotations. Higher numbers keep the GPU busy but consume more memory.
		bool use_local_mem_for_matching = false;				///< Enables / disables the local memory optimization.
		bool use_local_mem_for_erode = true;					///< Enables / disables the local memory optimization for the erode step applied to the texture mask.
	};
#endif
#ifdef TRLIB_TREE_MATCH_USE_OPENCL
	TreeMatchGPU(int min_patch_size, int patch_levels, double patch_quality_factor, int filter_resolution, double frequency_octaves, int num_filter_directions, const GPUMatchingOptions& gpu_matching_options = GPUMatchingOptions{});
#else
	TreeMatchGPU(int min_patch_size, int patch_levels, double patch_quality_factor, int filter_resolution, double frequency_octaves, int num_filter_directions);
#endif

#ifdef TRLIB_TREE_MATCH_USE_OPENCL
	static TreeMatchGPU load(const boost::filesystem::path& path, bool load_textures, const GPUMatchingOptions& gpu_matching_options = GPUMatchingOptions{});	
#else
	static TreeMatchGPU load(const boost::filesystem::path& path, bool load_textures);
#endif

	void add_target(const boost::filesystem::path& path, double dpi, double scale);
	void add_texture(const boost::filesystem::path& path, double dpi, double scale, int num_rotations, const TextureMarker& marker = TextureMarker(), const std::string id = std::string());
	void add_texture(const boost::filesystem::path& path, const boost::filesystem::path& mask, double dpi, double scale, int num_rotations, const TextureMarker& marker = TextureMarker(), const std::string id = std::string());

	void generate_patches(int target_index, const Grid& morphed_grid, cv::Mat edge_image);
	void generate_patches_square(int target_index);
	void add_patches(int target_index, const std::vector<PatchRegion>& patches, double scale);

	void compute_responses(double weight_intensity, double weight_sobel, double weight_gabor, double histogram_matching_factor);

	bool find_next_patch();
	bool find_next_patch_adaptive();

	cv::Mat fit_single_patch(const boost::filesystem::path& path);

	cv::Mat draw(int target_index, bool draw_target) const;
	cv::Mat draw_matched_target(int target_index, double histogram_matching_factor) const;
	cv::Mat draw_masked_target(int target_index) const;
	cv::Mat draw_patch(const Patch& patch) const;
	std::pair<cv::Mat, cv::Mat> draw_saliency(int target_index) const;

	std::vector<cv::Mat> draw_masked_textures() const;
	std::vector<cv::Mat> draw_masked_textures_patch(const std::vector<Patch>& patch, cv::Scalar color, double alpha) const;
	std::vector<cv::Mat> draw_masked_textures_patch(const std::vector<Patch>& patch, const std::vector<cv::Scalar>& color, const std::vector<double>& alpha) const;

	std::vector<cv::Mat> draw_masked_textures_patch(const Patch& patches, cv::Scalar color, double alpha) const;
	std::vector<cv::Mat> draw_masked_textures_patch_last(const std::vector<Patch>& patches, cv::Scalar color_1, double alpha_1, cv::Scalar color_2, double alpha_2, double scale) const;

	const std::vector<Texture>& targets() const
	{
		return m_targets;
	}

	const std::vector<std::vector<Texture>>& textures() const
	{
		return m_textures;
	}

	const GaborFilterBank& filter_bank() const
	{
		return m_filter_bank;
	}

	int num_targets() const
	{
		return static_cast<int>(m_targets.size());
	}

	int num_textures() const
	{
		return static_cast<int>(m_textures.size());
	}

	void downsample(int factor);

	void save(int target_index, boost::filesystem::path path) const;
	void find_markers(double marker_size_mm, int num_marker);

	cv::Size max_filter_size(double weight_intensity, double weight_sobel, double weight_gabor) const;

	const std::vector<Patch>& patches() const
	{
		return m_patches;
	}

	const std::deque<PatchRegion>& reconstruction_regions() const
	{
		return m_reconstruction_regions;
	}

	void sort_patches_by_saliency();
	void sort_patches_by_center_distance();

private:
	void mask_patch_resources(const Patch& patch);
	void mask_patch_resources(const AdaptivePatch& adaptive_patch);
	void mask_patch_resources(const Patch& patch, const cv::Mat& mask);

	void unmask_patch_resources(const Patch& patch);
	void unmask_patch_resources(const AdaptivePatch& adaptive_patch);
	void unmask_patch_resources(const Patch& patch, const cv::Mat& mask);

	void add_patch(const Patch& match);

	std::vector<Patch> match_patch(const PatchRegion& region);
	Patch match_patch_impl(const PatchRegion& region, cv::Mat mask);

	static cv::Mat compute_priority_map(const cv::Mat& texture);

	std::vector<cv::Size> m_patch_sizes;
	double m_patch_quality_factor;
	cv::Size m_subpatch_size;

	std::vector<Texture> m_targets;
	std::vector<std::vector<Texture>> m_textures;

	std::vector<cv::Mat> m_target_images;

	std::deque<PatchRegion> m_reconstruction_regions;

	GaborFilterBank m_filter_bank;
	std::vector<Patch> m_patches;

	// OpenCL Matcher
#ifdef TRLIB_TREE_MATCH_USE_OPENCL
	ocl_patch_matching::Matcher m_cl_matcher;
	std::size_t m_max_num_kernel_pixels_gpu;
#endif
};

#endif /* TRLIB_TREE_MATCH_GPU_HPP_ */