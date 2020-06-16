#include <ocl_template_matcher.hpp>
#include <ocl_wrappers.hpp>


// ----------------------------------------- IMPLEMENTATION ---------------------------------------------
namespace ocl_template_matching
{
	namespace impl
	{
		class MatcherImpl
		{
		public:
			MatcherImpl(const MatchingPolicyBase& matching_policy);
		private:
			cl::CLState m_cl_state;
		};

		MatcherImpl::MatcherImpl(const MatchingPolicyBase& matching_policy) :
			m_cl_state{matching_policy.platform_id(), matching_policy.device_id()}
		{
		}
	}
}


// ----------------------------------------- INTERFACE --------------------------------------------------

ocl_template_matching::impl::MatcherBase::MatcherBase(const MatchingPolicyBase& matching_policy) :
	m_impl(std::make_unique<MatcherImpl>(matching_policy))
{
	
}

ocl_template_matching::impl::MatcherBase::MatcherBase(MatcherBase&& other) noexcept = default;

ocl_template_matching::impl::MatcherBase& ocl_template_matching::impl::MatcherBase::operator=(MatcherBase&& other) noexcept = default;

ocl_template_matching::impl::MatcherBase::~MatcherBase()
{
}

ocl_template_matching::MatchingResult ocl_template_matching::impl::MatcherBase::match(const Texture& texture,
	const cv::Mat& texture_mask,
	const Texture& kernel,
	const cv::Mat& kernel_mask,
	const MatchingPolicyBase& matching_policy)
{
	return ocl_template_matching::MatchingResult{};
}

void ocl_template_matching::impl::MatcherBase::match(const Texture& texture,
	const cv::Mat& texture_mask,
	const Texture& kernel,
	const cv::Mat& kernel_mask,
	MatchingResult& result,
	const MatchingPolicyBase& matching_policy)
{

}