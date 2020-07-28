/** \file matching_policies.hpp
*	\author Fabian Friederichs
*
*	\brief Contains different implementations of matching policies for use with the Matcher class from ocl_patch_matcher.hpp.
*/

#ifndef _MATCHING_POLICIES_HPP_
#define _MATCHING_POLICIES_HPP_

#include <ocl_patch_matcher.hpp>
#include <memory>

namespace ocl_patch_matching
{
	/**
	 *	\namespace ocl_patch_matching::matching_policies
	 *	\brief Contains different implementations of matching policies for use with the Matcher class from ocl_patch_matcher.hpp.
	*/
	namespace matching_policies
	{
		namespace impl { class CLMatcherImpl; }
		/**
		 *	\brief Implements patch matching using OpenCL 1.2 GPU capabilities.
		*/
		class CLMatcher : public ocl_patch_matching::MatchingPolicyBase
		{
		public:
			/**
			 *	\brief	Defines the origin or anchor of the kernels.
			*/
			enum class ResultOrigin
			{
				UpperLeftCorner,	///< Returned match position refers to the upper left corner of the (possibly rotated) kernel, superimposed somewhere in the input texture.
				Center				///< Returned match position refers to the center of the (possibly rotated) kernel. The center pixel coordinate in the kernel is computed as (floor((width - 1) / 2), floor((height - 1) / 2).
			};

			/**
			 *	\brief Initializes a new instance of the CLMatcher matching policy.
			 *	\param max_texture_cache_memory			Maximum memory in bytes to use for caching input textures. Currently ignored.
			 *	\param local_block_size					Maximum desired local work group size (square 2D blocks! work group size in number of process elements is local_block_size^2). Due to local and private memory constraints, the local block size may be choosen smaller than this.
			 *	\param constant_kernel_max_pixels		Maximum number of kernel pixels for which the constant buffer optimization shall be applied.
			 *	\param local_buffer_max_pixels			Maximum number of input texture window pixels + kernel overlap for which the local buffer optimization shall be applied.
			 *	\param max_pipelined_matching_passes	Maximum number of rotations per batch. Higher numbers keep the GPU busy but increase memory footprint.
			 *	\param result_origin					Origin of the kernel. See ResultOrigin enum class.
			 *	\param use_local_mem_for_matching		Enable / disable local memory optimization for matching. Whether prefetching input texture pixels into local memory or not heavily depends on the device and specific kernel sizes.
			 *	\param use_local_mem_for_erode			Enable / disable local memory optimization for texture mask erosion.  Whether prefetching texture mask pixels into local memory or not heavily depends on the device and specific kernel sizes.
			*/
			CLMatcher(
				std::size_t max_texture_cache_memory,
				std::size_t local_block_size = 16,
				std::size_t constant_kernel_max_pixels = 50ull,
				std::size_t local_buffer_max_pixels = 64ull,
				std::size_t max_pipelined_matching_passes = 16ull,
				ResultOrigin result_origin = ResultOrigin::UpperLeftCorner,
				bool use_local_mem_for_matching = false,
				bool use_local_mem_for_erode = false
			);
			/// Destructor
			~CLMatcher() noexcept;

			/// No copies allowed.
			CLMatcher(const CLMatcher&) = delete;
			/// Default move constructor.
			CLMatcher(CLMatcher&&) noexcept = default;
			/// No copies allowed.
			CLMatcher& operator=(const CLMatcher&) = delete;
			/// Default move assignment.
			CLMatcher& operator=(CLMatcher&&) noexcept = default;

			/// Returns true because this class needs an OpenCL context for operation.
			bool uses_opencl() const override { return true; }
			/// Loads and compiles OpenCL kernels and creates initial resources.
			void initialize_opencl_state(const std::shared_ptr<simple_cl::cl::Context>& clcontext) override;
			/// Cleans up left over OpenCL state.
			void cleanup_opencl_state() override;

			/**
			 *	\brief                      Performs one matching pass given texture, kernel and a number of rotations.
			 *	\note	This implementation only returns the best match via MatchingResult.
			 *	\param texture              Input texture.
			 *	\param kernel               Kernel or template to be searched for in texture.
			 *	\param texture_rotations    Input texture rotations to try.
			 *	\param[out] match_res_out   Result of the matching pass.
			*/
			void compute_matches(
				const Texture& texture,
				const Texture& kernel,
				const std::vector<double>& texture_rotations,
				MatchingResult& match_res_out
			) override;

			/**
			 *  \brief                      Performs one matching pass given texture, kernel and a number of rotations. Possible matches are masked using texture_mask.
			 *	\note	This implementation only returns the best match via MatchingResult.
			 *  texture_mask can be any grayscale image. Every pixel in the input texture with the corresponsing mask pixel > 0 is considered as a potential match candidate.
			 *  As an optional step, the mask can be eroded with the kernel bounding box as structuring element, first.
			 *  \param texture              Input texture.
			 *  \param texture_mask         Input texture mask. Must be a grayscale (single channel!) image of the same dimensions as texture.
			 *  \param kernel               Kernel or template to be searched for in texture.
			 *  \param texture_rotations    Input texture rotations to try.
			 *  \param[out] match_res_out   Result of the matching pass
			 *  \param erode_texture_mask   If true, the texture mask is eroded with the kernel's bounding box as structuring element before use.
			*/
			void compute_matches(
				const Texture& texture,
				const cv::Mat& texture_mask,
				const Texture& kernel,
				const std::vector<double>& texture_rotations,
				MatchingResult& match_res_out,
				bool erode_texture_mask = true
			) override;

			/**
			 *  \brief                      Performs one matching pass given texture, kernel and a number of rotations. The kernel is masked using kernel_mask.
			 *	\note	This implementation only returns the best match via MatchingResult.
			 *  kernel_mask can be any grayscale image. Only kernel pixels whose corresponding kernel mask pixel is > 0 are considered for the calculation of matching costs.
			 *  As an optional step, the mask can be eroded with the kernel bounding box as structuring element, first.
			 *  \param texture              Input texture.
			 *  \param kernel               Kernel or template to be searched for in texture.
			 *  \param kernel_mask          Kernel mask. Must be a grayscale (single channel!) image of the same dimension as kernel.
			 *  \param texture_rotations    Input texture rotations to try.
			 *  \param[out] match_res_out   Result of the matching pass
			*/
			void compute_matches(
				const Texture& texture,
				const Texture& kernel,
				const cv::Mat& kernel_mask,
				const std::vector<double>& texture_rotations,
				MatchingResult& match_res_out
			) override;

			/**
			 *  \brief                      Performs one matching pass given texture, kernel and a number of rotations. Possible matches are masked using texture_mask and the kernel is masked using kernel_mask.
			 *	\note	This implementation only returns the best match via MatchingResult.
			 *  texture_mask can be any grayscale image. Every pixel in the input texture with the corresponsing mask pixel > 0 is considered as a potential match candidate.
			 *  kernel_mask can be any grayscale image. Only kernel pixels whose corresponding kernel mask pixel is > 0 are considered for the calculation of matching costs.
			 *  As an optional step, the mask can be eroded with the kernel bounding box as structuring element, first.
			 *  \param texture              Input texture.
			 *  \param texture_mask         Input texture mask. Must be a grayscale (single channel!) image of the same dimensions as texture.
			 *  \param kernel               Kernel or template to be searched for in texture.
			 *  \param kernel_mask          Kernel mask. Must be a grayscale (single channel!) image of the same dimension as kernel.
			 *  \param texture_rotations    Input texture rotations to try.
			 *  \param[out] match_res_out   Result of the matching pass
			 *  \param erode_texture_mask   If true, the texture mask is eroded with the kernel mask as structuring element before use.
			*/
			void compute_matches(
				const Texture& texture,
				const cv::Mat& texture_mask,
				const Texture& kernel,
				const cv::Mat& kernel_mask,
				const std::vector<double>& texture_rotations,
				MatchingResult& match_res_out,
				bool erode_texture_mask = true
			) override;

			/**
			 *  \brief                      Returns dimensions of the resulting cost matrix.
			 *	In this implementation, the cost matrix is a single channel gray scale image of type CV_32FC1. Width and height are the dimensions of the input texture minus the kernel overlap for the input rotation given by the parameter texture_rotation.
			 *  \param texture              Input texture.
			 *  \param kernel               Kernel texture.
			 *  \param texture_rotation     Input texture rotation angle to calculate the result dimensions for.
			 *  \return                     First and second component define width and height of the resulting cost matrix, the third component is always 1.
			*/
			cv::Vec3i response_dimensions(
				const Texture& texture,
				const Texture& kernel,
				double texture_rotation
			) const override;

			/**
			 *	\brief	Returns CV_32FC1, a single channel, single precision floating point image.
			 * @return cv::CV_32FC1
			*/
			match_response_cv_mat_t response_image_data_type(
				const Texture& texture,
				const Texture& kernel,
				double texture_rotation
			) const override;

		private:
			std::unique_ptr<impl::CLMatcherImpl> m_impl;					///< Pointer to implementation
			impl::CLMatcherImpl* impl() { return m_impl.get(); }			///< Accessor for implementing const correctness
			const impl::CLMatcherImpl* impl() const { return m_impl.get(); }///< Accessor for implementing const correctness
		};

		/**
		 *	\brief	Hybrid matching policy which chooses one of two policies based on input texture and kernel.
		 *
		 *	MatcherSelector must provide a function call operator: bool operator()(const Texture& input_texture, const Texture& kernel_texture). If true is returned, MatcherA is choosen, otherwise MatcherB.
		 *
		 *	\tparam MatcherA		Matching policy A.
		 *	\tparam MatcherB		Matching policy B.
		 *	\tparam MatcherSelector Predicate for choosing between A and B.
		*/
		template<typename MatcherA, typename MatcherB, typename MatcherSelector>
		class HybridMatcher : public ocl_patch_matching::MatchingPolicyBase
		{
		public:
			/**
			 *	\brief Initializes a new HybridMatcher instance.
			 *	\param matcher_a_instance	rvalue reference to a MatcherA instance.
			 *	\param matcher_b_instance	rvalue reference to a MatcherB instance.
			*/
			HybridMatcher(MatcherA&& matcher_a_instance, MatcherB&& matcher_b_instance) :
				m_matcher_a_instance(std::move(matcher_a_instance)),
				m_matcher_b_instance(std::move(matcher_b_instance))
			{
			}

			/// Destructor
			~HybridMatcher() noexcept
			{
			}

			/// No copies allowed.
			HybridMatcher(const HybridMatcher&) = delete;
			/// Default move constructor.
			HybridMatcher(HybridMatcher&&) noexcept = default;
			/// No copies allowed.
			HybridMatcher& operator=(const HybridMatcher&) = delete;
			/// Default move assignment.
			HybridMatcher& operator=(HybridMatcher&&) noexcept = default;

			/// If one of the two matching strategies uses OpenCL, this function returns true, false otherwise.
			bool uses_opencl() const override { return m_matcher_a_instance.uses_opencl || m_matcher_b_instance.uses_opencl }
			
			/// If uses_opencl() is true for MatcherA or MatcherB, the OpenCL context is forwarded to the corresponding functions.
			void initialize_opencl_state(const std::shared_ptr<simple_cl::cl::Context>& clcontext) override
			{
				if(m_matcher_a_instance.uses_opencl())
					m_matcher_a_instance.initialize_opencl_state(clcontext);
				if(m_matcher_b_instance.uses_opencl())
					m_matcher_b_instance.initialize_opencl_state(clcontext);
			}
			
			/// Forwards this call to MatcherA and MatcherB.
			void cleanup_opencl_state() override
			{
				if(m_matcher_a_instance.uses_opencl())
					m_matcher_a_instance.cleanup_opencl_state();
				if(m_matcher_b_instance.uses_opencl())
					m_matcher_b_instance.cleanup_opencl_state();
			}

			/**
			 *  \brief                      Performs one matching pass given texture, kernel and a number of rotations.
			 *
			 *	Uses MatcherSelector to choose between MatcherA's or MatcherB's implementation.
			 *
			 *  \param texture              Input texture.
			 *  \param kernel               Kernel or template to be searched for in texture.
			 *  \param texture_rotations    Input texture rotations to try.
			 *  \param[out] match_res_out   Result of the matching pass.
			*/
			void compute_matches(
				const Texture& texture,
				const Texture& kernel,
				const std::vector<double>& texture_rotations,
				MatchingResult& match_res_out
			) override
			{
				if(MatcherSelector(texture, kernel))
					m_matcher_a_instance.compute_matches(texture, kernel, texture_rotations, match_res_out);
				else
					m_matcher_b_instance.compute_matches(texture, kernel, texture_rotations, match_res_out);
			}

			/**
			 *  \brief                      Performs one matching pass given texture, kernel and a number of rotations. Possible matches are masked using texture_mask.
			 *  texture_mask can be any grayscale image. Every pixel in the input texture with the corresponsing mask pixel > 0 is considered as a potential match candidate.
			 *  As an optional step, the mask can be eroded with the kernel bounding box as structuring element, first.
			 *
			 *	Uses MatcherSelector to choose between MatcherA's or MatcherB's implementation.
			 *
			 *  \param texture              Input texture.
			 *  \param texture_mask         Input texture mask. Must be a grayscale (single channel!) image of the same dimensions as texture.
			 *  \param kernel               Kernel or template to be searched for in texture.
			 *  \param texture_rotations    Input texture rotations to try.
			 *  \param[out] match_res_out   Result of the matching pass
			 *  \param erode_texture_mask   If true, the texture mask is eroded with the kernel's bounding box as structuring element before use.
			*/
			void compute_matches(
				const Texture& texture,
				const cv::Mat& texture_mask,
				const Texture& kernel,
				const std::vector<double>& texture_rotations,
				MatchingResult& match_res_out,
				bool erode_texture_mask = true
			) override
			{
				if(MatcherSelector(texture, kernel))
					m_matcher_a_instance.compute_matches(texture, texture_mask, kernel, texture_rotations, match_res_out);
				else
					m_matcher_b_instance.compute_matches(texture, texture_mask, kernel, texture_rotations, match_res_out);
			}

			/**
			 *  \brief                      Performs one matching pass given texture, kernel and a number of rotations. The kernel is masked using kernel_mask.
			 *  kernel_mask can be any grayscale image. Only kernel pixels whose corresponding kernel mask pixel is > 0 are considered for the calculation of matching costs.
			 *  As an optional step, the mask can be eroded with the kernel bounding box as structuring element, first.
			 *	
			 *	Uses MatcherSelector to choose between MatcherA's or MatcherB's implementation.
			 *
			 *  \param texture              Input texture.
			 *  \param kernel               Kernel or template to be searched for in texture.
			 *  \param kernel_mask          Kernel mask. Must be a grayscale (single channel!) image of the same dimension as kernel.
			 *  \param texture_rotations    Input texture rotations to try.
			 *  \param[out] match_res_out   Result of the matching pass
			*/
			void compute_matches(
				const Texture& texture,
				const Texture& kernel,
				const cv::Mat& kernel_mask,
				const std::vector<double>& texture_rotations,
				MatchingResult& match_res_out
			) override
			{
				if(MatcherSelector(texture, kernel))
					m_matcher_a_instance.compute_matches(texture, kernel, kernel_mask, texture_rotations, match_res_out);
				else
					m_matcher_b_instance.compute_matches(texture, kernel, kernel_mask, texture_rotations, match_res_out);
			}

			/**
			 *  \brief                      Performs one matching pass given texture, kernel and a number of rotations. Possible matches are masked using texture_mask and the kernel is masked using kernel_mask.
			 *  texture_mask can be any grayscale image. Every pixel in the input texture with the corresponsing mask pixel > 0 is considered as a potential match candidate.
			 *  kernel_mask can be any grayscale image. Only kernel pixels whose corresponding kernel mask pixel is > 0 are considered for the calculation of matching costs.
			 *  As an optional step, the mask can be eroded with the kernel bounding box as structuring element, first.
			 *	
			 *	Uses MatcherSelector to choose between MatcherA's or MatcherB's implementation.
			 *
			 *  \param texture              Input texture.
			 *  \param texture_mask         Input texture mask. Must be a grayscale (single channel!) image of the same dimensions as texture.
			 *  \param kernel               Kernel or template to be searched for in texture.
			 *  \param kernel_mask          Kernel mask. Must be a grayscale (single channel!) image of the same dimension as kernel.
			 *  \param texture_rotations    Input texture rotations to try.
			 *  \param[out] match_res_out   Result of the matching pass
			 *  \param erode_texture_mask   If true, the texture mask is eroded with the kernel mask as structuring element before use.
			*/
			void compute_matches(
				const Texture& texture,
				const cv::Mat& texture_mask,
				const Texture& kernel,
				const cv::Mat& kernel_mask,
				const std::vector<double>& texture_rotations,
				MatchingResult& match_res_out,
				bool erode_texture_mask = true
			) override
			{
				if(MatcherSelector(texture, kernel))
					m_matcher_a_instance.compute_matches(texture, texture_mask, kernel, kernel_mask, texture_rotations, match_res_out);
				else
					m_matcher_b_instance.compute_matches(texture, texture_mask, kernel, kernel_mask, texture_rotations, match_res_out);
			}

			/**
			 *  \brief                      Returns the dimensions of the resulting cost matrix given some texture, kernel and rotation angle in radians.
			 *
			 *	Uses MatcherSelector to choose between MatcherA's or MatcherB's implementation.
			 *
			 *  \param texture              Texture instance for which the result dimensions shall be calculated.
			 *  \param kernel               Kernel texture instance in case the output dimension also depends on the kernel size.
			 *  \param texture_rotation     Texture rotation angle to calculate the result dimensions for.
			 *  \return                     Returns a cv::Vec3i. First and second component define width and height of the resulting cost matrix, the third component tells the number of channels the response will have.
			*/
			cv::Vec3i response_dimensions(
				const Texture& texture,
				const Texture& kernel,
				double texture_rotation
			) const override
			{
				if(MatcherSelector(texture, kernel))
					m_matcher_a_instance.response_dimensions(texture, kernel, texture_rotation);
				else
					m_matcher_b_instance.response_dimensions(texture, kernel, texture_rotation);
			}

			/**
			 *  \brief                      Returns the OpenCV datatype used in the resulting cost matrix (e.g. CV_32FC1).
			 *
			 *	Uses MatcherSelector to choose between MatcherA's or MatcherB's implementation.
			 *
			 *  \param texture              Input texture.
			 *  \param kernel               Kernel texture instance.
			 *  \param texture_rotation     Texture rotation.
			 *  \return                     OpenCV datatype id.
			*/
			match_response_cv_mat_t response_image_data_type(
				const Texture& texture,
				const Texture& kernel,
				double texture_rotation
			) const override
			{
				if(MatcherSelector(texture, kernel))
					m_matcher_a_instance.response_image_data_type(texture, kernel, texture_rotation);
				else
					m_matcher_b_instance.response_image_data_type(texture, kernel, texture_rotation);
			}

		private:
			MatcherA m_matcher_a_instance; ///< MatcherA instance
			MatcherB m_matcher_b_instance; ///< MatcherB instance
		};
	}
}
#endif