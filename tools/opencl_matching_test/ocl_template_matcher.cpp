#include <ocl_template_matcher.hpp>
#include <simple_cl.hpp>


// ----------------------------------------- IMPLEMENTATION ---------------------------------------------
namespace ocl_template_matching
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
					m_context = simple_cl::cl::Context::createInstance(m_matching_policy->platform_id(), m_matching_policy->device_id());
			};

			void match(const Texture& texture, const cv::Mat& texture_mask, const Texture& kernel, const cv::Mat& kernel_mask, double texture_rotation, MatchingResult& result)
			{
				if(m_matching_policy->uses_opencl())
				{
					// calculate response
					m_matching_policy->compute_response(texture, texture_mask, kernel, kernel_mask, texture_rotation, result, m_context.get());
					// extract best matches
					m_matching_policy->find_best_matches(result, m_context.get());
				}
				else
				{
					// calculate response
					m_matching_policy->compute_response(texture, texture_mask, kernel, kernel_mask, texture_rotation, result);
					// extract best matches
					m_matching_policy->find_best_matches(result);
				}
			};
		private:
			MatchingPolicyBase* m_matching_policy;
			std::shared_ptr<simple_cl::cl::Context> m_context;
		};
	}
}


// ----------------------------------------- INTERFACE --------------------------------------------------

ocl_template_matching::Matcher::Matcher(std::unique_ptr<MatchingPolicyBase>&& matching_policy) :
	m_impl(new impl::MatcherImpl(matching_policy.get())),
	m_matching_policy(std::move(matching_policy))
{

}

ocl_template_matching::Matcher::Matcher(Matcher&& other) noexcept :
	m_matching_policy(std::move(other.m_matching_policy)),
	m_impl(std::move(other.m_impl))
{
}

ocl_template_matching::Matcher& ocl_template_matching::Matcher::operator=(Matcher&& other) noexcept
{
	if(this == &other) return *this;

	std::swap(m_matching_policy, other.m_matching_policy);
	std::swap(m_impl, other.m_impl);

	return *this;
}

ocl_template_matching::Matcher::~Matcher() noexcept
{
}

ocl_template_matching::MatchingResult ocl_template_matching::Matcher::match(const Texture& texture, const cv::Mat& texture_mask, const Texture& kernel, const cv::Mat& kernel_mask, double texture_rotation)
{
	MatchingResult result;
	impl()->match(texture, texture_mask, kernel, kernel_mask, texture_rotation, result);
	return result;
}

void ocl_template_matching::Matcher::match(const Texture& texture, const cv::Mat& texture_mask, const Texture& kernel, const cv::Mat& kernel_mask, double texture_rotation, MatchingResult& result)
{
	impl()->match(texture, texture_mask, kernel, kernel_mask, texture_rotation, result);
}
