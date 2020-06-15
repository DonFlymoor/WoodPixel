#ifndef _MATCHING_POLICIES_HPP_
#define _MATCHING_POLICIES_HPP_

#include <ocl_template_matcher.hpp>

// default policy for testing
class DefaultMatchingPolicy : public ocl_template_matching::MatchingPolicyBase
{
public:
	DefaultMatchingPolicy(std::size_t platform_id_, std::size_t device_id_) :
		MatchingPolicyBase(),
		m_platform_id(platform_id_),
		m_device_id(device_id_)
	{
	}

	std::size_t platform_id() const override
	{
		return m_platform_id;
	}

	std::size_t device_id() const override
	{
		return m_device_id;
	}

private:
	const std::size_t m_platform_id;
	const std::size_t m_device_id;
};

#endif