#include <matching_policies.hpp>
#include <simple_cl.hpp>
#include <opencv2/opencv.hpp>

namespace ocl_template_matching
{
	namespace matching_policies
	{
		namespace impl
		{
			#pragma region CLMatcherImpl declaration
			// include the kernel files
			#include <kernels/kernel_sqdiff_naive.hpp>

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

				void initialize_opencl_state(const std::shared_ptr<simple_cl::cl::Context>& clcontext);
				void cleanup_opencl_state();

				void compute_response(
					const Texture& texture,
					const cv::Mat& texture_mask,
					const Texture& kernel,
					const cv::Mat& kernel_mask,
					double texture_rotation,
					MatchingResult& match_res_out
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
				static simple_cl::cl::Image::ImageDesc make_input_image_desc(const Texture& input_tex);
				static simple_cl::cl::Image::ImageDesc make_output_image_desc(const Texture& input_tex, const Texture& kernel_tex);
				static simple_cl::cl::Image::ImageDesc make_kernel_image_desc(const Texture& kernel_tex);
				static simple_cl::cl::Image::ImageDesc make_mask_image_desc(const cv::Mat& texture_mask);
				static simple_cl::cl::Image::ImageDesc make_kernel_mask_image_desc(const cv::Mat& kernel_mask);

				static cv::Vec3i get_response_dimensions(const Texture& texture, const Texture& kernel);

				ocl_template_matching::matching_policies::CLMatcher::DeviceSelectionPolicy m_selection_policy;
				std::size_t m_max_tex_cache_size;

				// TODO: Think about texture cache!

				// OpenCL context
				std::shared_ptr<simple_cl::cl::Context> m_cl_context;

				// OpenCL programs
				std::unique_ptr<simple_cl::cl::Program> m_program_naive_sqdiff;

				// Kernel handles
				simple_cl::cl::Program::CLKernelHandle m_kernel_naive_sqdiff;
			};
			#pragma endregion

			#pragma region CLMatcherImpl implementation
			// static functions
			inline cv::Vec3i ocl_template_matching::matching_policies::impl::CLMatcherImpl::get_response_dimensions(const Texture& texture, const Texture& kernel)
			{
				return cv::Vec3i{
					texture.response.cols() - kernel.response.cols() + 1,
					texture.response.rows() - kernel.response.rows() + 1,
					1
				};
			}

			simple_cl::cl::Image::ImageDesc ocl_template_matching::matching_policies::impl::CLMatcherImpl::make_input_image_desc(const Texture& input_tex)
			{
				return simple_cl::cl::Image::ImageDesc{
					simple_cl::cl::Image::ImageType::Image2DArray,	// One array slice per response channel
					simple_cl::cl::Image::ImageDimensions{
						static_cast<std::size_t>(input_tex.response.cols()),				// width
						static_cast<std::size_t>(input_tex.response.rows()),				// height
						static_cast<std::size_t>(input_tex.response.num_channels())			// number of slices
					},
					simple_cl::cl::Image::ImageChannelOrder::R,		// One red channel per slice
					simple_cl::cl::Image::ImageChannelType::FLOAT,	// Single precision floating point data
					simple_cl::cl::MemoryFlags{
						simple_cl::cl::DeviceAccess::ReadOnly,		// Kernel may only read
						simple_cl::cl::HostAccess::WriteOnly,		// Host may only write
						simple_cl::cl::HostPointerOption::None		// No host pointer stuff
					},
					simple_cl::cl::Image::HostPitch{
						0ull,										// No host pointer is given so host pitch is ignored
						0ull
					},
					nullptr											// no host pointer
				};
			}

			simple_cl::cl::Image::ImageDesc ocl_template_matching::matching_policies::impl::CLMatcherImpl::make_output_image_desc(const Texture& input_tex, const Texture& kernel_tex)
			{
				auto output_dims{get_response_dimensions(input_tex, kernel_tex)};
				return simple_cl::cl::Image::ImageDesc{
					simple_cl::cl::Image::ImageType::Image2D,		// One response channel
					simple_cl::cl::Image::ImageDimensions{
						static_cast<std::size_t>(output_dims[0]),				// width
						static_cast<std::size_t>(output_dims[1]),				// height
						1ull													// number of slices
					},
					simple_cl::cl::Image::ImageChannelOrder::R,		// One red channel
					simple_cl::cl::Image::ImageChannelType::FLOAT,	// Single precision floating point data
					simple_cl::cl::MemoryFlags{
						simple_cl::cl::DeviceAccess::WriteOnly,		// Kernel may only write
						simple_cl::cl::HostAccess::ReadOnly,		// Host may only read
						simple_cl::cl::HostPointerOption::None		// No host pointer stuff
					},
					simple_cl::cl::Image::HostPitch{
						0ull,										// No host pointer is given so host pitch is ignored
						0ull
					},
					nullptr											// no host pointer
				};
			}

			simple_cl::cl::Image::ImageDesc ocl_template_matching::matching_policies::impl::CLMatcherImpl::make_kernel_image_desc(const Texture& kernel_tex)
			{
				return simple_cl::cl::Image::ImageDesc{
					simple_cl::cl::Image::ImageType::Image2DArray,	// One array slice per response channel
					simple_cl::cl::Image::ImageDimensions{
						static_cast<std::size_t>(kernel_tex.response.cols()),				// width
						static_cast<std::size_t>(kernel_tex.response.rows()),				// height
						static_cast<std::size_t>(kernel_tex.response.num_channels())			// number of slices
					},
					simple_cl::cl::Image::ImageChannelOrder::R,		// One red channel per slice
					simple_cl::cl::Image::ImageChannelType::FLOAT,	// Single precision floating point data
					simple_cl::cl::MemoryFlags{
						simple_cl::cl::DeviceAccess::ReadOnly,		// Kernel may only read
						simple_cl::cl::HostAccess::WriteOnly,		// Host may only write
						simple_cl::cl::HostPointerOption::None		// No host pointer stuff
					},
					simple_cl::cl::Image::HostPitch{
						0ull,										// No host pointer is given so host pitch is ignored
						0ull
					},
					nullptr											// no host pointer
				};
			}

			simple_cl::cl::Image::ImageDesc ocl_template_matching::matching_policies::impl::CLMatcherImpl::make_mask_image_desc(const cv::Mat& texture_mask)
			{
				return simple_cl::cl::Image::ImageDesc{
					simple_cl::cl::Image::ImageType::Image2D,		// One response channel
					simple_cl::cl::Image::ImageDimensions{
						static_cast<std::size_t>(texture_mask.cols),				// width
						static_cast<std::size_t>(texture_mask.rows),				// height
						1ull														// number of slices
					},
					simple_cl::cl::Image::ImageChannelOrder::R,		// One red channel
					simple_cl::cl::Image::ImageChannelType::FLOAT,	// Single precision floating point data
					simple_cl::cl::MemoryFlags{
						simple_cl::cl::DeviceAccess::ReadOnly,		// Kernel may only write
						simple_cl::cl::HostAccess::WriteOnly,		// Host may only read
						simple_cl::cl::HostPointerOption::None		// No host pointer stuff
					},
					simple_cl::cl::Image::HostPitch{
						0ull,										// No host pointer is given so host pitch is ignored
						0ull
					},
					nullptr											// no host pointer
				};
			}

			simple_cl::cl::Image::ImageDesc ocl_template_matching::matching_policies::impl::CLMatcherImpl::make_kernel_mask_image_desc(const cv::Mat& kernel_mask)
			{
				return simple_cl::cl::Image::ImageDesc{
					simple_cl::cl::Image::ImageType::Image2D,		// One response channel
					simple_cl::cl::Image::ImageDimensions{
						static_cast<std::size_t>(kernel_mask.cols),				// width
						static_cast<std::size_t>(kernel_mask.rows),				// height
						1ull													// number of slices
					},
					simple_cl::cl::Image::ImageChannelOrder::R,		// One red channel
					simple_cl::cl::Image::ImageChannelType::FLOAT,	// Single precision floating point data
					simple_cl::cl::MemoryFlags{
						simple_cl::cl::DeviceAccess::ReadOnly,		// Kernel may only write
						simple_cl::cl::HostAccess::WriteOnly,		// Host may only read
						simple_cl::cl::HostPointerOption::None		// No host pointer stuff
					},
					simple_cl::cl::Image::HostPitch{
						0ull,										// No host pointer is given so host pitch is ignored
						0ull
					},
					nullptr											// no host pointer
				};
			}

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

			inline void ocl_template_matching::matching_policies::impl::CLMatcherImpl::select_platform_and_device(std::size_t& platform_idx, std::size_t& device_idx) const
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

			inline void ocl_template_matching::matching_policies::impl::CLMatcherImpl::initialize_opencl_state(const std::shared_ptr<simple_cl::cl::Context>& clcontext)
			{
				// save context
				m_cl_context = clcontext;
				// create and compile programs
				m_program_naive_sqdiff.reset(new simple_cl::cl::Program(kernels::sqdiff_naive_src, kernels::sqdiff_naive_copt, m_cl_context));
				// retrieve kernel handles
				m_kernel_naive_sqdiff = m_program_naive_sqdiff->getKernel("sqdiff_naive");
			}

			inline void ocl_template_matching::matching_policies::impl::CLMatcherImpl::cleanup_opencl_state()
			{
			}

			inline void ocl_template_matching::matching_policies::impl::CLMatcherImpl::compute_response(
				const Texture& texture,
				const cv::Mat& texture_mask,
				const Texture& kernel,
				const cv::Mat& kernel_mask,
				double texture_rotation,
				MatchingResult& match_res_out)
			{
				// TODO: Implement first stupid version of cl template matching, then optimize and add texture cache

				// Input and output images
				auto input_image_desc{make_input_image_desc(texture)};
				auto output_image_desc{make_output_image_desc(texture, kernel)};
				auto kernel_image_desc{make_kernel_image_desc(kernel)};
				auto tex_mask_image_desc{make_mask_image_desc(texture_mask)};
				auto kernel_mask_image_desc{make_kernel_mask_image_desc(kernel_mask)};

				simple_cl::cl::Image input_image(m_cl_context, input_image_desc);
				simple_cl::cl::Image output_image(m_cl_context, output_image_desc);
				simple_cl::cl::Image kernel_image(m_cl_context, kernel_image_desc);
				simple_cl::cl::Image tex_mask_image(m_cl_context, tex_mask_image_desc);
				simple_cl::cl::Image kernel_mask_image(m_cl_context, input_image_desc);

				// upload data

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
				return get_response_dimensions(texture, kernel);
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
			#pragma endregion
		}
	}
}

#pragma region CLMatcher interface

// ----------------------------------------------------------------------- INTERFACE -----------------------------------------------------------------------

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

void ocl_template_matching::matching_policies::CLMatcher::initialize_opencl_state(const std::shared_ptr<simple_cl::cl::Context>& clcontext)
{
	impl()->initialize_opencl_state(clcontext);
}

void ocl_template_matching::matching_policies::CLMatcher::cleanup_opencl_state()
{
	impl()->cleanup_opencl_state();
}

void ocl_template_matching::matching_policies::CLMatcher::compute_response(
	const Texture& texture,
	const cv::Mat& texture_mask,
	const Texture& kernel,
	const cv::Mat& kernel_mask,
	double texture_rotation,
	MatchingResult& match_res_out)
{
	impl()->compute_response(texture, texture_mask, kernel, kernel_mask, texture_rotation, match_res_out);
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
#pragma endregion