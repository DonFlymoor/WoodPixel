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
			MatcherImpl(MatchingPolicyBase* matching_policy, Matcher::DeviceSelectionPolicy device_selection_policy) :
				m_matching_policy(matching_policy),
				m_context(nullptr)
			{
				if(m_matching_policy->uses_opencl())
				{
					std::size_t plat_id, dev_id;
					select_platform_and_device(plat_id, dev_id, device_selection_policy);
					m_context = simple_cl::cl::Context::createInstance(plat_id, dev_id);
					m_matching_policy->initialize_opencl_state(m_context);
				}
			}

			~MatcherImpl()
			{
			}

			void match(const Texture& texture, const Texture& kernel, const std::vector<double>& texture_rotations, MatchingResult& result)
			{				
				// calculate response
				m_matching_policy->compute_matches(texture, kernel, texture_rotations, result);			
			};

			void match(const Texture& texture, const cv::Mat& texture_mask, const Texture& kernel, const std::vector<double>& texture_rotations, MatchingResult& result, bool erode_texture_mask)
			{
				// calculate response
				m_matching_policy->compute_matches(texture, texture_mask, kernel, texture_rotations, result, erode_texture_mask);
			};

			void match(const Texture& texture, const Texture& kernel, const cv::Mat& kernel_mask, const std::vector<double>& texture_rotations, MatchingResult& result)
			{
				// calculate response
				m_matching_policy->compute_matches(texture, kernel, kernel_mask, texture_rotations, result);
			};

			void match(const Texture& texture, const cv::Mat& texture_mask, const Texture& kernel, const cv::Mat& kernel_mask, const std::vector<double>& texture_rotations, MatchingResult& result, bool erode_texture_mask)
			{
				// calculate response
				m_matching_policy->compute_matches(texture, texture_mask, kernel, kernel_mask, texture_rotations, result, erode_texture_mask);
			};

			void select_platform_and_device(std::size_t& platform_idx, std::size_t& device_idx, Matcher::DeviceSelectionPolicy device_selection_policy) const
			{
				auto pdevinfo = simple_cl::cl::Context::read_platform_and_device_info();
				std::size_t plat_idx{0ull};
				std::size_t dev_idx{0ull};

				if(device_selection_policy == Matcher::DeviceSelectionPolicy::FirstSuitableDevice)
				{
					platform_idx = plat_idx;
					device_idx = dev_idx;
					return;
				}

				for(std::size_t p = 0; p < pdevinfo.size(); ++p)
				{
					for(std::size_t d = 0; d < pdevinfo[p].devices.size(); ++d)
					{
						switch(device_selection_policy)
						{
							case Matcher::DeviceSelectionPolicy::MostComputeUnits:
								if(pdevinfo[p].devices[d].max_compute_units > pdevinfo[plat_idx].devices[dev_idx].max_compute_units)
								{
									plat_idx = p;
									dev_idx = d;
								}
								break;
							case Matcher::DeviceSelectionPolicy::MostGPUThreads:
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

		private:

			MatchingPolicyBase* m_matching_policy;
			std::shared_ptr<simple_cl::cl::Context> m_context;
		};
	}
}


// ----------------------------------------- INTERFACE --------------------------------------------------

ocl_patch_matching::Matcher::Matcher(std::unique_ptr<MatchingPolicyBase>&& matching_policy, DeviceSelectionPolicy device_selection_policy) :
	m_matching_policy(std::move(matching_policy)),
	m_impl(new impl::MatcherImpl(matching_policy.get(), device_selection_policy))	
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

void ocl_patch_matching::Matcher::match(const Texture& texture, const Texture& kernel, const std::vector<double>& texture_rotations, MatchingResult& result)
{
	impl()->match(texture, kernel, texture_rotations, result);
}

void ocl_patch_matching::Matcher::match(const Texture& texture, const cv::Mat& texture_mask, const Texture& kernel, const std::vector<double>& texture_rotations, MatchingResult& result, bool erode_texture_mask)
{
	impl()->match(texture, texture_mask, kernel, texture_rotations, result, erode_texture_mask);
}

void ocl_patch_matching::Matcher::match(const Texture& texture, const Texture& kernel, const cv::Mat& kernel_mask, const std::vector<double>& texture_rotations, MatchingResult& result)
{
	impl()->match(texture, kernel, kernel_mask, texture_rotations, result);
}

void ocl_patch_matching::Matcher::match(const Texture& texture, const cv::Mat& texture_mask, const Texture& kernel, const cv::Mat& kernel_mask, const std::vector<double>& texture_rotations, MatchingResult& result, bool erode_texture_mask)
{
	impl()->match(texture, texture_mask, kernel, kernel_mask, texture_rotations, result, erode_texture_mask);
}

void ocl_patch_matching::Matcher::match(const Texture& texture, const Texture& kernel, double texture_rotation, MatchingResult& result)
{
	static std::vector<double> rots(1, 0.0);
	rots[0] = texture_rotation;
	impl()->match(texture, kernel, rots, result);
}

void ocl_patch_matching::Matcher::match(const Texture& texture, const cv::Mat& texture_mask, const Texture& kernel, double texture_rotation, MatchingResult& result, bool erode_texture_mask)
{
	static std::vector<double> rots(1, 0.0);
	rots[0] = texture_rotation;
	impl()->match(texture, texture_mask, kernel, rots, result, erode_texture_mask);
}

void ocl_patch_matching::Matcher::match(const Texture& texture, const Texture& kernel, const cv::Mat& kernel_mask, double texture_rotation, MatchingResult& result)
{
	static std::vector<double> rots(1, 0.0);
	rots[0] = texture_rotation;
	impl()->match(texture, kernel, kernel_mask, rots, result);
}

void ocl_patch_matching::Matcher::match(const Texture& texture, const cv::Mat& texture_mask, const Texture& kernel, const cv::Mat& kernel_mask, double texture_rotation, MatchingResult& result, bool erode_texture_mask)
{
	static std::vector<double> rots(1, 0.0);
	rots[0] = texture_rotation;
	impl()->match(texture, texture_mask, kernel, kernel_mask, rots, result, erode_texture_mask);
}
