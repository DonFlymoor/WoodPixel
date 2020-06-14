#include <ocl_template_matcher.hpp>
#include <CL/cl.h>
#include <ocl_error.hpp>


// ----------------------------------------- IMPLEMENTATION ---------------------------------------------

namespace ocl_template_matching
{
	namespace impl
	{
		namespace cl
		{
			class OpenCLState
			{
			public:
				OpenCLState(std::size_t platform_index, std::size_t device_index, bool gpu = true)
				{
					// -------------------------- setup OpenCL --------------------------------------------------
					try
					{
						cl_int cl_error{CL_SUCCESS};
						// query number of platforms available
						cl_uint number_of_platforms{0};
						cl_error = CL_EX(clGetPlatformIDs(0, nullptr, &number_of_platforms));
						// query platform ID's
						std::unique_ptr<cl_platform_id[]> platform_ids(new cl_platform_id[number_of_platforms]);
						// store platform info
						for(std::size_t p = 0; p < static_cast<std::size_t>(number_of_platforms); ++p)
						{

						}
					}
					catch(...)
					{
						throw;
					}
				}

			private:
				// Hold list of platforms and devices for convenience
				struct CLPlatform { cl_platform_id id; std::vector<cl_device_id> devices; };
				std::vector<CLPlatform> m_available_platforms;

				// ID's and handles for current OpenCL instance
				cl_platform_id m_current_platform;
				cl_device_id m_current_device;
				cl_context m_current_context;
				cl_command_queue m_current_command_queue;
			};


		}

		

		class MatcherImpl
		{
		public:
			MatcherImpl(const MatchingStrategyBase& matching_strat);
		private:
			cl::OpenCLState m_cl_state;
		};

		MatcherImpl::MatcherImpl(const MatchingStrategyBase& matching_strat) :
			m_cl_state{}
		{
		}
	}
}


// ----------------------------------------- INTERFACE --------------------------------------------------

ocl_template_matching::impl::MatcherBase::MatcherBase(const MatchingStrategyBase& matching_strat) :
	m_impl(std::make_unique<MatcherImpl>(matching_strat))
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
	const MatchingStrategyBase& matching_strat)
{
	return ocl_template_matching::MatchingResult{};
}

void ocl_template_matching::impl::MatcherBase::match(const Texture& texture,
	const cv::Mat& texture_mask,
	const Texture& kernel,
	const cv::Mat& kernel_mask,
	MatchingResult& result,
	const MatchingStrategyBase& matching_strat)
{

}