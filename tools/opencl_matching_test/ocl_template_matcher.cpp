#include <ocl_template_matcher.hpp>


// ----------------------------------------- IMPLEMENTATION ---------------------------------------------

namespace ocl_template_matching
{
	namespace impl
	{
		class MatcherImpl
		{

		};
	}
}


// ----------------------------------------- INTERFACE --------------------------------------------------

ocl_template_matching::impl::MatcherBase::MatcherBase(const MatchingStrategyBase& matching_strat,
	const MatchExtractionStrategyBase& matchex_strat) :
	m_impl(nullptr)
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
	const MatchingStrategyBase& matching_strat,
	const MatchExtractionStrategyBase& matchex_strat)
{
	return ocl_template_matching::MatchingResult{};
}

void ocl_template_matching::impl::MatcherBase::match(const Texture& texture,
	const cv::Mat& texture_mask,
	const Texture& kernel,
	const cv::Mat& kernel_mask,
	MatchingResult& result,
	const MatchingStrategyBase& matching_strat,
	const MatchExtractionStrategyBase& matchex_strat)
{

}