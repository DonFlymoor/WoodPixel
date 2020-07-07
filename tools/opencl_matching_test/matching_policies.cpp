#include <matching_policies.hpp>
#include <simple_cl.hpp>
#include <opencv2/opencv.hpp>

namespace ocl_template_matching
{
	namespace matching_policies
	{
		namespace impl
		{
			class CLMatcherImpl
			{
			public:
				CLMatcherImpl(ocl_template_matching::matching_policies::CLMatcher::DeviceSelectionPolicy device_selection_policy, std::size_t max_texture_cache_memory);
				~CLMatcherImpl() noexcept;
				CLMatcherImpl(const CLMatcherImpl&) = delete;
				CLMatcherImpl(CLMatcherImpl&&) = delete;
				CLMatcherImpl& operator=(const CLMatcherImpl&) = delete;
				CLMatcherImpl& operator=(CLMatcherImpl&&) = delete;

				std::size_t platform_id() const;
				std::size_t device_id() const;

				void compute_response(
					const Texture& texture,
					const cv::Mat& texture_mask,
					const Texture& kernel,
					const cv::Mat& kernel_mask,
					double texture_rotation,
					MatchingResult& match_res_out,
					simple_cl::cl::Context* clcontext
				);

				void find_best_matches(MatchingResult& match_res_out);

				cv::Vec3i response_dimensions(
					const Texture& texture,
					const cv::Mat& texture_mask,
					const Texture& kernel,
					const cv::Mat& kernel_mask,
					double texture_rotation
				) const;

				match_response_cv_mat_t response_image_data_type(
					const Texture& texture,
					const cv::Mat& texture_mask,
					const Texture& kernel,
					const cv::Mat& kernel_mask,
					double texture_rotation
				) const;

			private:
				void select_platform_and_device(std::size_t& platform_idx, std::size_t& device_idx) const;

				ocl_template_matching::matching_policies::CLMatcher::DeviceSelectionPolicy m_selection_policy;
				std::size_t m_max_tex_cache_size;

				// TODO: Think about texture cache!
			};

			inline ocl_template_matching::matching_policies::impl::CLMatcherImpl::CLMatcherImpl(
				ocl_template_matching::matching_policies::CLMatcher::DeviceSelectionPolicy device_selection_policy,
				std::size_t max_texture_cache_memory) :
				m_selection_policy(device_selection_policy),
				m_max_tex_cache_size(max_texture_cache_memory)				
			{
			}

			inline ocl_template_matching::matching_policies::impl::CLMatcherImpl::~CLMatcherImpl() noexcept
			{
			}

			void ocl_template_matching::matching_policies::impl::CLMatcherImpl::select_platform_and_device(std::size_t& platform_idx, std::size_t& device_idx) const
			{
				auto pdevinfo = simple_cl::cl::Context::read_platform_and_device_info();
				std::size_t plat_idx{0ull};
				std::size_t dev_idx{0ull};

				if(m_selection_policy == ocl_template_matching::matching_policies::CLMatcher::DeviceSelectionPolicy::FirstSuitableDevice)
				{
					platform_idx = plat_idx;
					device_idx = dev_idx;
					return;
				}

				for(std::size_t p = 0; p < pdevinfo.size(); ++p)
				{
					for(std::size_t d = 0; d < pdevinfo[p].devices.size(); ++d)
					{
						switch(m_selection_policy)
						{
							case ocl_template_matching::matching_policies::CLMatcher::DeviceSelectionPolicy::MostComputeUnits:
								if(pdevinfo[p].devices[d].max_compute_units > pdevinfo[plat_idx].devices[dev_idx].max_compute_units) 
								{ 
									plat_idx = p;
									dev_idx = d;
								}
								break;
							case ocl_template_matching::matching_policies::CLMatcher::DeviceSelectionPolicy::MostGPUThreads:
								if(pdevinfo[p].devices[d].max_compute_units * pdevinfo[p].devices[d].max_work_group_size > pdevinfo[plat_idx].devices[dev_idx].max_compute_units * pdevinfo[plat_idx].devices[dev_idx].max_work_group_size)
								{
									plat_idx = p;
									dev_idx = d;
								}
								break;
							default:
								break;
						}
					}
				}

				platform_idx = plat_idx;
				device_idx = dev_idx;
			}

			inline std::size_t ocl_template_matching::matching_policies::impl::CLMatcherImpl::platform_id() const
			{
				std::size_t pidx, didx;
				select_platform_and_device(pidx, didx);
				return pidx;
			}

			inline std::size_t ocl_template_matching::matching_policies::impl::CLMatcherImpl::device_id() const
			{
				std::size_t pidx, didx;
				select_platform_and_device(pidx, didx);
				return didx;
			}

			inline void ocl_template_matching::matching_policies::impl::CLMatcherImpl::compute_response(
				const Texture& texture,
				const cv::Mat& texture_mask,
				const Texture& kernel,
				const cv::Mat& kernel_mask,
				double texture_rotation,
				MatchingResult& match_res_out,
				simple_cl::cl::Context* clcontext)
			{
				// TODO: Implement first stupid version of cl template matching, then optimize and add texture cache
			}

			
			inline void ocl_template_matching::matching_policies::impl::CLMatcherImpl::find_best_matches(MatchingResult& match_res_out)
			{
				// TODO: Implement parallel extraction of best match
			}

			inline cv::Vec3i ocl_template_matching::matching_policies::impl::CLMatcherImpl::response_dimensions(
				const Texture& texture,
				const cv::Mat& texture_mask,
				const Texture& kernel,
				const cv::Mat& kernel_mask,
				double texture_rotation) const
			{
				return cv::Vec3i{
					texture.response.cols() - kernel.response.cols() + 1,
					texture.response.rows() - kernel.response.rows() + 1,
					1
				};
			}

			inline match_response_cv_mat_t ocl_template_matching::matching_policies::impl::CLMatcherImpl::response_image_data_type(
				const Texture& texture,
				const cv::Mat& texture_mask,
				const Texture& kernel,
				const cv::Mat& kernel_mask,
				double texture_rotation) const
			{
				return CV_32FC1;
			}
		}
	}
}

// class CLMatcher
ocl_template_matching::matching_policies::CLMatcher::CLMatcher(DeviceSelectionPolicy device_selection_policy, std::size_t max_texture_cache_memory) :
	m_impl(new impl::CLMatcherImpl(device_selection_policy, max_texture_cache_memory))
{
}

ocl_template_matching::matching_policies::CLMatcher::~CLMatcher() noexcept
{
}

std::size_t ocl_template_matching::matching_policies::CLMatcher::platform_id() const
{
	return impl()->platform_id();
}

std::size_t ocl_template_matching::matching_policies::CLMatcher::device_id() const
{
	return impl()->device_id();
}

void ocl_template_matching::matching_policies::CLMatcher::compute_response(
	const Texture& texture,
	const cv::Mat& texture_mask,
	const Texture& kernel,
	const cv::Mat& kernel_mask,
	double texture_rotation,
	MatchingResult& match_res_out,
	simple_cl::cl::Context* clcontext)
{
	impl()->compute_response(texture, texture_mask, kernel, kernel_mask, texture_rotation, match_res_out, clcontext);
}

void ocl_template_matching::matching_policies::CLMatcher::find_best_matches(MatchingResult& match_res_out)
{
	impl()->find_best_matches(match_res_out);
}

cv::Vec3i ocl_template_matching::matching_policies::CLMatcher::response_dimensions(
	const Texture& texture,
	const cv::Mat& texture_mask,
	const Texture& kernel,
	const cv::Mat& kernel_mask,
	double texture_rotation) const
{
	return impl()->response_dimensions(texture, texture_mask, kernel, kernel_mask, texture_rotation);
}

ocl_template_matching::match_response_cv_mat_t ocl_template_matching::matching_policies::CLMatcher::response_image_data_type(
	const Texture& texture,
	const cv::Mat& texture_mask,
	const Texture& kernel,
	const cv::Mat& kernel_mask,
	double texture_rotation) const
{
	return impl()->response_image_data_type(texture, texture_mask, kernel, kernel_mask, texture_rotation);
}
