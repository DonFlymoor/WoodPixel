#include <ocl_patch_matcher.hpp>
#include <simple_cl.hpp>


// ----------------------------------------- IMPLEMENTATION ---------------------------------------------
namespace ocl_patch_matching
{
	namespace impl
	{
		class MatcherImpl
		{
		public:
			MatcherImpl(MatchingPolicyBase* matching_policy) :
				m_matching_policy(matching_policy),
				m_context(nullptr)
			{
				if(m_matching_policy->uses_opencl())
				{
					m_context = simple_cl::cl::Context::createInstance(m_matching_policy->platform_id(), m_matching_policy->device_id());
					m_matching_policy->initialize_opencl_state(m_context);
				}
			}

			~MatcherImpl()
			{
			}

			void erode_texture_mask(const cv::Mat& texture_mask, cv::Mat& texture_mask_eroded, const cv::Mat& kernel_mask, const cv::Point& kernel_anchor, double texture_rotation)
			{
				m_matching_policy->erode_texture_mask(texture_mask, texture_mask_eroded, kernel_mask, kernel_anchor, texture_rotation);
			}

			void match(const Texture& texture, const Texture& kernel, double texture_rotation, MatchingResult& result)
			{				
				// calculate response
				m_matching_policy->compute_matches(texture, kernel, texture_rotation, result);			
			};

			void match(const Texture& texture, const cv::Mat& texture_mask, const Texture& kernel, double texture_rotation, MatchingResult& result)
			{
				// calculate response
				m_matching_policy->compute_matches(texture, texture_mask, kernel, texture_rotation, result);
			};

			void match(const Texture& texture, const Texture& kernel, const cv::Mat& kernel_mask, double texture_rotation, MatchingResult& result)
			{
				// calculate response
				m_matching_policy->compute_matches(texture, kernel, kernel_mask, texture_rotation, result);
			};

			void match(const Texture& texture, const cv::Mat& texture_mask, const Texture& kernel, const cv::Mat& kernel_mask, double texture_rotation, MatchingResult& result)
			{
				// calculate response
				m_matching_policy->compute_matches(texture, texture_mask, kernel, kernel_mask, texture_rotation, result);
			};

		private:

			MatchingPolicyBase* m_matching_policy;
			std::shared_ptr<simple_cl::cl::Context> m_context;
		};
	}
}


// ----------------------------------------- INTERFACE --------------------------------------------------

ocl_patch_matching::Matcher::Matcher(std::unique_ptr<MatchingPolicyBase>&& matching_policy) :
	m_matching_policy(std::move(matching_policy)),
	m_impl(new impl::MatcherImpl(matching_policy.get()))
	
{
}

ocl_patch_matching::Matcher::Matcher(Matcher&& other) noexcept :
	m_matching_policy(std::move(other.m_matching_policy)),
	m_impl(std::move(other.m_impl))
{
}

ocl_patch_matching::Matcher& ocl_patch_matching::Matcher::operator=(Matcher&& other) noexcept
{
	if(this == &other) return *this;

	std::swap(m_matching_policy, other.m_matching_policy);
	std::swap(m_impl, other.m_impl);

	return *this;
}

ocl_patch_matching::Matcher::~Matcher() noexcept
{
}

void ocl_patch_matching::Matcher::erode_texture_mask(const cv::Mat& texture_mask, cv::Mat& texture_mask_eroded, const cv::Mat& kernel_mask, const cv::Point& kernel_anchor, double texture_rotation)
{
	impl()->erode_texture_mask(texture_mask, texture_mask_eroded, kernel_mask, kernel_anchor, texture_rotation);
}

void ocl_patch_matching::Matcher::match(const Texture& texture, const Texture& kernel, double texture_rotation, MatchingResult& result)
{
	impl()->match(texture, kernel, texture_rotation, result);
}

void ocl_patch_matching::Matcher::match(const Texture& texture, const cv::Mat& texture_mask, const Texture& kernel, double texture_rotation, MatchingResult& result)
{
	impl()->match(texture, texture_mask, kernel, texture_rotation, result);
}

void ocl_patch_matching::Matcher::match(const Texture& texture, const Texture& kernel, const cv::Mat& kernel_mask, double texture_rotation, MatchingResult& result)
{
	impl()->match(texture, kernel, kernel_mask, texture_rotation, result);
}

void ocl_patch_matching::Matcher::match(const Texture& texture, const cv::Mat& texture_mask, const Texture& kernel, const cv::Mat& kernel_mask, double texture_rotation, MatchingResult& result)
{
	impl()->match(texture, texture_mask, kernel, kernel_mask, texture_rotation, result);
}
