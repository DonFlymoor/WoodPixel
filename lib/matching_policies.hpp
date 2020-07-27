#ifndef _MATCHING_POLICIES_HPP_
#define _MATCHING_POLICIES_HPP_

#include <ocl_patch_matcher.hpp>
#include <memory>

namespace ocl_patch_matching
{
	namespace matching_policies
	{
		// Implements template matching via OpenCL capabilities
		namespace impl { class CLMatcherImpl; }
		class CLMatcher : public ocl_patch_matching::MatchingPolicyBase
		{
		public:
			enum class ResultOrigin
			{
				UpperLeftCorner,
				Center
			};

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
			~CLMatcher() noexcept;

			CLMatcher(const CLMatcher&) = delete;
			CLMatcher(CLMatcher&&) noexcept = default;
			CLMatcher& operator=(const CLMatcher&) = delete;
			CLMatcher& operator=(CLMatcher&&) noexcept = default;

			bool uses_opencl() const override { return true; }
			void initialize_opencl_state(const std::shared_ptr<simple_cl::cl::Context>& clcontext) override;
			void cleanup_opencl_state() override;

			void compute_matches(
				const Texture& texture,
				const Texture& kernel,
				const std::vector<double>& texture_rotations,
				MatchingResult& match_res_out
			) override;

			void compute_matches(
				const Texture& texture,
				const cv::Mat& texture_mask,
				const Texture& kernel,
				const std::vector<double>& texture_rotations,
				MatchingResult& match_res_out,
				bool erode_texture_mask = true
			) override;

			void compute_matches(
				const Texture& texture,
				const Texture& kernel,
				const cv::Mat& kernel_mask,
				const std::vector<double>& texture_rotations,
				MatchingResult& match_res_out
			) override;

			void compute_matches(
				const Texture& texture,
				const cv::Mat& texture_mask,
				const Texture& kernel,
				const cv::Mat& kernel_mask,
				const std::vector<double>& texture_rotations,
				MatchingResult& match_res_out,
				bool erode_texture_mask = true
			) override;

			cv::Vec3i response_dimensions(
				const Texture& texture,
				const Texture& kernel,
				double texture_rotation
			) const override;

			match_response_cv_mat_t response_image_data_type(
				const Texture& texture,
				const Texture& kernel,
				double texture_rotation
			) const override;

		private:
			std::unique_ptr<impl::CLMatcherImpl> m_impl;
			impl::CLMatcherImpl* impl() { return m_impl.get(); }
			const impl::CLMatcherImpl* impl() const { return m_impl.get(); }
		};

		// Hybrid approach which chooses between two matchers based on the template size
		template<typename MatcherA, typename MatcherB, typename MatcherSelector>
		class HybridMatcher : public ocl_patch_matching::MatchingPolicyBase
		{
		public:
			HybridMatcher(MatcherA&& matcher_a_instance, MatcherB&& matcher_b_instance) :
				m_matcher_a_instance(std::move(matcher_a_instance)),
				m_matcher_b_instance(std::move(matcher_b_instance))
			{

			}

			~HybridMatcher() noexcept
			{

			}

			HybridMatcher(const HybridMatcher&) = delete;
			HybridMatcher(HybridMatcher&&) noexcept = default;
			HybridMatcher& operator=(const HybridMatcher&) = delete;
			HybridMatcher& operator=(HybridMatcher&&) noexcept = default;

			bool uses_opencl() const override { return m_matcher_a_instance.uses_opencl || m_matcher_b_instance.uses_opencl }
			
			void initialize_opencl_state(const std::shared_ptr<simple_cl::cl::Context>& clcontext) override
			{
				if(m_matcher_a_instance.uses_opencl())
					m_matcher_a_instance.initialize_opencl_state(clcontext);
				if(m_matcher_b_instance.uses_opencl())
					m_matcher_b_instance.initialize_opencl_state(clcontext);
			}
			
			void cleanup_opencl_state() override
			{
				if(m_matcher_a_instance.uses_opencl())
					m_matcher_a_instance.cleanup_opencl_state();
				if(m_matcher_b_instance.uses_opencl())
					m_matcher_b_instance.cleanup_opencl_state();
			}

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
			MatcherA m_matcher_a_instance;
			MatcherB m_matcher_b_instance;
		};
	}
}
#endif