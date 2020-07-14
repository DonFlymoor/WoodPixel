#ifndef _MATCHING_POLICIES_HPP_
#define _MATCHING_POLICIES_HPP_

#include <ocl_template_matcher.hpp>
#include <memory>

namespace ocl_template_matching
{
	namespace matching_policies
	{
		// Implements template matching via OpenCL capabilities
		namespace impl { class CLMatcherImpl; }
		class CLMatcher : public ocl_template_matching::MatchingPolicyBase
		{
		public:
			enum class DeviceSelectionPolicy
			{
				MostComputeUnits,
				MostGPUThreads,
				FirstSuitableDevice
			};

			CLMatcher(DeviceSelectionPolicy device_selection_policy, std::size_t max_texture_cache_memory, std::size_t local_block_size = 16, std::size_t constant_kernel_max_pixels = 50ull);
			~CLMatcher() noexcept;

			std::size_t platform_id() const override;
			std::size_t device_id() const override;

			bool uses_opencl() const override { return true; }
			void initialize_opencl_state(const std::shared_ptr<simple_cl::cl::Context>& clcontext) override;
			void cleanup_opencl_state() override;

			void compute_response(
				const Texture& texture,
				const Texture& kernel,
				double texture_rotation,
				MatchingResult& match_res_out
			) override;

			void compute_response(
				const Texture& texture,
				const Texture& kernel,
				const cv::Mat& kernel_mask,
				double texture_rotation,
				MatchingResult& match_res_out
			) override;

			void find_best_matches(MatchingResult& match_res_out) override;
			void find_best_matches(MatchingResult& match_res_out, const cv::Mat& texture_mask) override;

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

		//// Uses OpenCV for template matching
		//class OpenCVMatcher : public ocl_template_matching::MatchingPolicyBase
		//{

		//};

		//// Hybrid approach which chooses between two matchers based on the template size
		//template<typename MatcherA, typename MatcherB>
		//class HybridMatcher : public ocl_template_matching::MatchingPolicyBase
		//{

		//};
	}
}
#endif