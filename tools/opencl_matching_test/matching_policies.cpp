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
					const Texture& kernel,
					double texture_rotation,
					MatchingResult& match_res_out
				);

				void compute_response(
					const Texture& texture,
					const cv::Mat& texture_mask,
					const Texture& kernel,
					double texture_rotation,
					MatchingResult& match_res_out
				);

				void compute_response(
					const Texture& texture,
					const Texture& kernel,
					const cv::Mat& kernel_mask,
					double texture_rotation,
					MatchingResult& match_res_out
				);

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
					const Texture& kernel,
					double texture_rotation
				) const;

				

				match_response_cv_mat_t response_image_data_type(
					const Texture& texture,
					const Texture& kernel,
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

				// non blocking
				void upload_texture(const Texture& fv, simple_cl::cl::Image& climage, std::vector<simple_cl::cl::Event>& events);
				// blocking
				void upload_texture(const Texture& fv, simple_cl::cl::Image& climage);
				// non blocking
				void upload_mask(const cv::Mat& mask, simple_cl::cl::Image& climage, std::vector<simple_cl::cl::Event>& events);
				// blocking
				void upload_mask(const cv::Mat& mask, simple_cl::cl::Image& climage);

				void prepare_output_image(const Texture& input, const Texture& kernel);
				simple_cl::cl::Event clear_output_image(float value = std::numeric_limits<float>::max());

				simple_cl::cl::Event read_output_image(cv::Mat& out_mat, const cv::Vec3i& output_size, const std::vector<simple_cl::cl::Event>& wait_for);

				ocl_template_matching::matching_policies::CLMatcher::DeviceSelectionPolicy m_selection_policy;
				std::size_t m_max_tex_cache_size;

				// TODO: Think about texture cache!

				// Output buffer. Only use a single output image and enlarge it when necessary.
				std::unique_ptr<simple_cl::cl::Image> m_output_buffer;

				// OpenCL context
				std::shared_ptr<simple_cl::cl::Context> m_cl_context;

				// OpenCL programs
				std::unique_ptr<simple_cl::cl::Program> m_program_naive_sqdiff;

				// Kernel handles
				simple_cl::cl::Program::CLKernelHandle m_kernel_naive_sqdiff_both_masks;
				simple_cl::cl::Program::CLKernelHandle m_kernel_naive_sqdiff_tex_mask;
				simple_cl::cl::Program::CLKernelHandle m_kernel_naive_sqdiff_kernel_mask;
				simple_cl::cl::Program::CLKernelHandle m_kernel_naive_sqdiff_no_mask;
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

			inline simple_cl::cl::Image::ImageDesc ocl_template_matching::matching_policies::impl::CLMatcherImpl::make_input_image_desc(const Texture& input_tex)
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
						simple_cl::cl::HostAccess::ReadWrite,		// Host may only write // TODO: dbg
						simple_cl::cl::HostPointerOption::None		// No host pointer stuff
					},
					simple_cl::cl::Image::HostPitch{
						0ull,										// No host pointer is given so host pitch is ignored
						0ull
					},
					nullptr											// no host pointer
				};
			}

			inline simple_cl::cl::Image::ImageDesc ocl_template_matching::matching_policies::impl::CLMatcherImpl::make_output_image_desc(const Texture& input_tex, const Texture& kernel_tex)
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

			inline simple_cl::cl::Image::ImageDesc ocl_template_matching::matching_policies::impl::CLMatcherImpl::make_kernel_image_desc(const Texture& kernel_tex)
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
						simple_cl::cl::HostAccess::ReadWrite,		// Host may only write
						simple_cl::cl::HostPointerOption::None		// No host pointer stuff
					},
					simple_cl::cl::Image::HostPitch{
						0ull,										// No host pointer is given so host pitch is ignored
						0ull
					},
					nullptr											// no host pointer
				};
			}

			inline simple_cl::cl::Image::ImageDesc ocl_template_matching::matching_policies::impl::CLMatcherImpl::make_mask_image_desc(const cv::Mat& texture_mask)
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

			inline simple_cl::cl::Image::ImageDesc ocl_template_matching::matching_policies::impl::CLMatcherImpl::make_kernel_mask_image_desc(const cv::Mat& kernel_mask)
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
				m_kernel_naive_sqdiff_both_masks = m_program_naive_sqdiff->getKernel("sqdiff_naive_both_masks");
				m_kernel_naive_sqdiff_tex_mask = m_program_naive_sqdiff->getKernel("sqdiff_naive_tex_mask");
				m_kernel_naive_sqdiff_kernel_mask = m_program_naive_sqdiff->getKernel("sqdiff_naive_kernel_mask");
				m_kernel_naive_sqdiff_no_mask = m_program_naive_sqdiff->getKernel("sqdiff_naive_no_mask");
			}

			inline void ocl_template_matching::matching_policies::impl::CLMatcherImpl::cleanup_opencl_state()
			{
			}

			void display_image(const std::string& name, const cv::Mat& mat, bool wait = false)
			{
				cv::imshow(name, mat);
				if(wait)
					cv::waitKey();
			}

			void ocl_template_matching::matching_policies::impl::CLMatcherImpl::upload_texture(const Texture& fv, simple_cl::cl::Image& climage, std::vector<simple_cl::cl::Event>& events)
			{
				// image region
				simple_cl::cl::Image::ImageRegion region{
					simple_cl::cl::Image::ImageOffset{0ull, 0ull, 0ull},
					simple_cl::cl::Image::ImageDimensions{static_cast<std::size_t>(fv.response.size().width), static_cast<std::size_t>(fv.response.size().height), 1ull}
				};
				// host format
				simple_cl::cl::Image::HostFormat hostfmt{
					simple_cl::cl::Image::HostChannelOrder{1ull, {simple_cl::cl::Image::ColorChannel::R, simple_cl::cl::Image::ColorChannel::R, simple_cl::cl::Image::ColorChannel::R, simple_cl::cl::Image::ColorChannel::R}},
					simple_cl::cl::Image::HostDataType::FLOAT,
					simple_cl::cl::Image::HostPitch{}
				};

				// iterate reponse channels, convert them to a float image and then copy into device memory
				// TODO: FIX THIS SHIT
				for(std::size_t i = 0; i < fv.response.num_channels(); ++i)
				{
					cv::Mat float_feature;
					fv.response[static_cast<int>(i)].convertTo(float_feature, CV_32FC1, 1.0 / 65535.0);
					// just to be safe, update row_pitch. I don't trust OpenCV's voodoo here.
					hostfmt.pitch.row_pitch = static_cast<std::size_t>(float_feature.step[0]);
					// select slice
					region.offset.offset_depth = i;
					// write texture data
					//events.push_back(std::move(climage.write(region, hostfmt, float_feature.data, true)));
					climage.write(region, hostfmt, float_feature.data, true);	
					display_image("write: " + std::to_string(i), float_feature, true);
				}

				for(std::size_t i = 0; i < fv.response.num_channels(); ++i)
				{
					region.offset.offset_depth = i;
					cv::Mat readdata(fv.response[i].rows, fv.response[i].cols, CV_32FC1);
					climage.read(region, hostfmt, readdata.data, true);
					display_image("read: " + std::to_string(i), readdata, true);
				}
			}

			void ocl_template_matching::matching_policies::impl::CLMatcherImpl::upload_texture(const Texture& fv, simple_cl::cl::Image& climage)
			{
				static std::vector<simple_cl::cl::Event> events;
				events.clear();
				upload_texture(fv, climage, events);
				simple_cl::cl::wait_for_events(events.begin(), events.end());
			}

			void ocl_template_matching::matching_policies::impl::CLMatcherImpl::upload_mask(const cv::Mat& mask, simple_cl::cl::Image& climage, std::vector<simple_cl::cl::Event>& events)
			{
				cv::Mat float_mask;
				mask.convertTo(float_mask, CV_32FC1);

				// image region
				simple_cl::cl::Image::ImageRegion region{
					simple_cl::cl::Image::ImageOffset{0ull, 0ull, 0ull},
					simple_cl::cl::Image::ImageDimensions{static_cast<std::size_t>(mask.cols), static_cast<std::size_t>(mask.rows), 1ull}
				};
				// host format
				simple_cl::cl::Image::HostFormat hostfmt{
					simple_cl::cl::Image::HostChannelOrder{1ull, {simple_cl::cl::Image::ColorChannel::R, simple_cl::cl::Image::ColorChannel::R, simple_cl::cl::Image::ColorChannel::R, simple_cl::cl::Image::ColorChannel::R}},
					simple_cl::cl::Image::HostDataType::FLOAT,
					simple_cl::cl::Image::HostPitch{static_cast<std::size_t>(float_mask.step[0]), 0ull}
				};
				// enqueue upload and return event
				events.push_back(std::move(climage.write(region, hostfmt, float_mask.data, true)));
			}

			void ocl_template_matching::matching_policies::impl::CLMatcherImpl::upload_mask(const cv::Mat& mask, simple_cl::cl::Image& climage)
			{
				cv::Mat float_mask;
				mask.convertTo(float_mask, CV_32FC1);

				// image region
				simple_cl::cl::Image::ImageRegion region{
					simple_cl::cl::Image::ImageOffset{0ull, 0ull, 0ull},
					simple_cl::cl::Image::ImageDimensions{static_cast<std::size_t>(mask.cols), static_cast<std::size_t>(mask.rows), 1ull}
				};
				// host format
				simple_cl::cl::Image::HostFormat hostfmt{
					simple_cl::cl::Image::HostChannelOrder{1ull, {simple_cl::cl::Image::ColorChannel::R, simple_cl::cl::Image::ColorChannel::R, simple_cl::cl::Image::ColorChannel::R, simple_cl::cl::Image::ColorChannel::R}},
					simple_cl::cl::Image::HostDataType::FLOAT,
					simple_cl::cl::Image::HostPitch{static_cast<std::size_t>(float_mask.step[0]), 0ull}
				};
				// enqueue upload and return event
				climage.write(region, hostfmt, float_mask.data, true).wait();
			}

			void ocl_template_matching::matching_policies::impl::CLMatcherImpl::prepare_output_image(const Texture& input, const Texture& kernel)
			{
				auto dims{get_response_dimensions(input, kernel)};
				if(m_output_buffer) // if output image already exists
				{
					// recreate output image only if it is too small for the new input - kernel combination
					if(static_cast<std::size_t>(dims[0]) > m_output_buffer->width() || static_cast<std::size_t>(dims[1]) > m_output_buffer->height() || static_cast<std::size_t>(dims[2]) > m_output_buffer->layers())
					{
						auto output_desc{make_output_image_desc(input, kernel)};
						m_output_buffer.reset(new simple_cl::cl::Image(m_cl_context, output_desc));
					}
				}
				else
				{
					auto output_desc{make_output_image_desc(input, kernel)};
					m_output_buffer.reset(new simple_cl::cl::Image(m_cl_context, output_desc));
				}
			}

			simple_cl::cl::Event CLMatcherImpl::clear_output_image(float value)
			{
				simple_cl::cl::Image::ImageRegion region{
					simple_cl::cl::Image::ImageOffset{0ull, 0ull, 0ull},
					simple_cl::cl::Image::ImageDimensions{m_output_buffer->width(), m_output_buffer->height(), m_output_buffer->layers()}
				};
				return m_output_buffer->fill(simple_cl::cl::Image::FillColor{value}, region);
			}

			simple_cl::cl::Event ocl_template_matching::matching_policies::impl::CLMatcherImpl::read_output_image(cv::Mat& out_mat, const cv::Vec3i& output_size, const std::vector<simple_cl::cl::Event>& wait_for)
			{
				// resize output if necessary
				if((output_size[0] != out_mat.cols) ||
					(output_size[1] != out_mat.rows) ||
					(output_size[2] != out_mat.channels()))
				{
					out_mat = cv::Mat(output_size[1], output_size[0], CV_32FC1);
				}

				// image region
				simple_cl::cl::Image::ImageRegion region{
					simple_cl::cl::Image::ImageOffset{0ull, 0ull, 0ull},
					simple_cl::cl::Image::ImageDimensions{static_cast<std::size_t>(output_size[0]), static_cast<std::size_t>(output_size[1]), 1ull}
				};

				// host format
				simple_cl::cl::Image::HostFormat hostfmt{
					simple_cl::cl::Image::HostChannelOrder{1ull, {simple_cl::cl::Image::ColorChannel::R, simple_cl::cl::Image::ColorChannel::R, simple_cl::cl::Image::ColorChannel::R, simple_cl::cl::Image::ColorChannel::R}},
					simple_cl::cl::Image::HostDataType::FLOAT,
					simple_cl::cl::Image::HostPitch{static_cast<std::size_t>(out_mat.step[0]), 0ull}
				};

				// read output and return event
				return m_output_buffer->read(region, hostfmt, out_mat.data, wait_for.begin(), wait_for.end());
			}

			inline void ocl_template_matching::matching_policies::impl::CLMatcherImpl::compute_response(
				const Texture& texture,
				const cv::Mat& texture_mask,
				const Texture& kernel,
				const cv::Mat& kernel_mask,
				double texture_rotation,
				MatchingResult& match_res_out)
			{
				static std::vector<simple_cl::cl::Event> pre_compute_events;
				pre_compute_events.clear();
				// Input and output images // TODO: cache these device textures somewhere. Allocating and filling them for every patch is pretty expensive.
				// create image descriptors
				auto input_image_desc{make_input_image_desc(texture)};
				auto kernel_image_desc{make_kernel_image_desc(kernel)};
				auto tex_mask_image_desc{make_mask_image_desc(texture_mask)};
				auto kernel_mask_image_desc{make_kernel_mask_image_desc(kernel_mask)};
				// create images
				simple_cl::cl::Image input_image(m_cl_context, input_image_desc);
				simple_cl::cl::Image kernel_image(m_cl_context, kernel_image_desc);
				simple_cl::cl::Image tex_mask_image(m_cl_context, tex_mask_image_desc);
				simple_cl::cl::Image kernel_mask_image(m_cl_context, kernel_mask_image_desc);
				// upload data
				upload_texture(texture, input_image, pre_compute_events);
				upload_texture(kernel, kernel_image, pre_compute_events);
				upload_mask(texture_mask, tex_mask_image, pre_compute_events);
				upload_mask(kernel_mask, kernel_mask_image, pre_compute_events);
				// output dims
				auto out_dims{get_response_dimensions(texture, kernel)};
				// allocate output buffer if needed
				prepare_output_image(texture, kernel);
				// run kernel
				// execution params
				simple_cl::cl::Program::ExecParams exec_params{
					2ull,
					{0ull, 0ull, 0ull},
					{static_cast<std::size_t>(out_dims[0]), static_cast<std::size_t>(out_dims[1]), 1ull},
					{8ull, 8ull, 1ull}
				};
				simple_cl::cl::Event compute_finished = (*m_program_naive_sqdiff)(
					m_kernel_naive_sqdiff_both_masks,
					pre_compute_events.begin(), pre_compute_events.end(),
					exec_params,
					input_image,
					kernel_image,
					tex_mask_image,
					kernel_mask_image,
					*m_output_buffer,
					cl_int2{static_cast<int>(input_image_desc.dimensions.width), static_cast<int>(input_image_desc.dimensions.height)},
					cl_int2{static_cast<int>(kernel_image_desc.dimensions.width), static_cast<int>(kernel_image_desc.dimensions.height)},
					cl_float2{std::sinf(static_cast<float>(texture_rotation)), std::cosf(static_cast<float>(texture_rotation))}
				);
				// read result
				
				pre_compute_events.clear();
				pre_compute_events.push_back(std::move(compute_finished));
				read_output_image(match_res_out.total_cost_matrix, out_dims, pre_compute_events).wait();
			}

			inline void ocl_template_matching::matching_policies::impl::CLMatcherImpl::compute_response(
				const Texture& texture,
				const cv::Mat& texture_mask,
				const Texture& kernel,
				double texture_rotation,
				MatchingResult& match_res_out)
			{
				static std::vector<simple_cl::cl::Event> pre_compute_events;
				pre_compute_events.clear();
				// Input and output images // TODO: cache these device textures somewhere. Allocating and filling them for every patch is pretty expensive.
				// create image descriptors
				auto input_image_desc{make_input_image_desc(texture)};
				auto kernel_image_desc{make_kernel_image_desc(kernel)};
				auto tex_mask_image_desc{make_mask_image_desc(texture_mask)};
				// create images
				simple_cl::cl::Image input_image(m_cl_context, input_image_desc);
				simple_cl::cl::Image kernel_image(m_cl_context, kernel_image_desc);
				simple_cl::cl::Image tex_mask_image(m_cl_context, tex_mask_image_desc);
				// upload data
				upload_texture(texture, input_image, pre_compute_events);
				upload_texture(kernel, kernel_image, pre_compute_events);
				upload_mask(texture_mask, tex_mask_image, pre_compute_events);
				// output dims
				auto out_dims{get_response_dimensions(texture, kernel)};
				// allocate output buffer if needed
				prepare_output_image(texture, kernel);
				// run kernel
				// execution params
				simple_cl::cl::Program::ExecParams exec_params{
					2ull,
					{0ull, 0ull, 0ull},
					{static_cast<std::size_t>(out_dims[0]), static_cast<std::size_t>(out_dims[1]), 1ull},
					{8ull, 8ull, 1ull}
				};
				simple_cl::cl::Event compute_finished = (*m_program_naive_sqdiff)(
					m_kernel_naive_sqdiff_tex_mask,
					pre_compute_events.begin(), pre_compute_events.end(),
					exec_params,
					input_image,
					kernel_image,
					tex_mask_image,
					*m_output_buffer,
					cl_int2{static_cast<int>(input_image_desc.dimensions.width), static_cast<int>(input_image_desc.dimensions.height)},
					cl_int2{static_cast<int>(kernel_image_desc.dimensions.width), static_cast<int>(kernel_image_desc.dimensions.height)},
					cl_float2{std::sinf(static_cast<float>(texture_rotation)), std::cosf(static_cast<float>(texture_rotation))}
					);
					// read result

				pre_compute_events.clear();
				pre_compute_events.push_back(std::move(compute_finished));
				read_output_image(match_res_out.total_cost_matrix, out_dims, pre_compute_events).wait();
			}

			inline void ocl_template_matching::matching_policies::impl::CLMatcherImpl::compute_response(
				const Texture& texture,
				const Texture& kernel,
				const cv::Mat& kernel_mask,
				double texture_rotation,
				MatchingResult& match_res_out)
			{
				static std::vector<simple_cl::cl::Event> pre_compute_events;
				pre_compute_events.clear();
				// Input and output images // TODO: cache these device textures somewhere. Allocating and filling them for every patch is pretty expensive.
				// create image descriptors
				auto input_image_desc{make_input_image_desc(texture)};
				auto kernel_image_desc{make_kernel_image_desc(kernel)};
				auto kernel_mask_image_desc{make_kernel_mask_image_desc(kernel_mask)};
				// create images
				simple_cl::cl::Image input_image(m_cl_context, input_image_desc);
				simple_cl::cl::Image kernel_image(m_cl_context, kernel_image_desc);
				simple_cl::cl::Image kernel_mask_image(m_cl_context, kernel_mask_image_desc);
				// upload data
				upload_texture(texture, input_image, pre_compute_events);
				upload_texture(kernel, kernel_image, pre_compute_events);
				upload_mask(kernel_mask, kernel_mask_image, pre_compute_events);
				// output dims
				auto out_dims{get_response_dimensions(texture, kernel)};
				// allocate output buffer if needed
				prepare_output_image(texture, kernel);
				// run kernel
				// execution params
				simple_cl::cl::Program::ExecParams exec_params{
					2ull,
					{0ull, 0ull, 0ull},
					{static_cast<std::size_t>(out_dims[0]), static_cast<std::size_t>(out_dims[1]), 1ull},
					{8ull, 8ull, 1ull}
				};
				simple_cl::cl::Event compute_finished = (*m_program_naive_sqdiff)(
					m_kernel_naive_sqdiff_kernel_mask,
					pre_compute_events.begin(), pre_compute_events.end(),
					exec_params,
					input_image,
					kernel_image,
					kernel_mask_image,
					*m_output_buffer,
					cl_int2{static_cast<int>(input_image_desc.dimensions.width), static_cast<int>(input_image_desc.dimensions.height)},
					cl_int2{static_cast<int>(kernel_image_desc.dimensions.width), static_cast<int>(kernel_image_desc.dimensions.height)},
					cl_float2{std::sinf(static_cast<float>(texture_rotation)), std::cosf(static_cast<float>(texture_rotation))}
					);
					// read result

				pre_compute_events.clear();
				pre_compute_events.push_back(std::move(compute_finished));
				read_output_image(match_res_out.total_cost_matrix, out_dims, pre_compute_events).wait();
			}

			inline void ocl_template_matching::matching_policies::impl::CLMatcherImpl::compute_response(
				const Texture& texture,
				const Texture& kernel,
				double texture_rotation,
				MatchingResult& match_res_out)
			{
				static std::vector<simple_cl::cl::Event> pre_compute_events;
				pre_compute_events.clear();
				// Input and output images // TODO: cache these device textures somewhere. Allocating and filling them for every patch is pretty expensive.
				// create image descriptors
				auto input_image_desc{make_input_image_desc(texture)};
				auto kernel_image_desc{make_kernel_image_desc(kernel)};
				// create images
				simple_cl::cl::Image input_image(m_cl_context, input_image_desc);
				simple_cl::cl::Image kernel_image(m_cl_context, kernel_image_desc);
				// upload data
				upload_texture(texture, input_image, pre_compute_events);
				upload_texture(kernel, kernel_image, pre_compute_events);
				// output dims
				auto out_dims{get_response_dimensions(texture, kernel)};
				// allocate output buffer if needed
				prepare_output_image(texture, kernel);
				// run kernel
				// execution params
				simple_cl::cl::Program::ExecParams exec_params{
					2ull,
					{0ull, 0ull, 0ull},
					{static_cast<std::size_t>(out_dims[0]), static_cast<std::size_t>(out_dims[1]), 1ull},
					{8ull, 8ull, 1ull}
				};
				simple_cl::cl::Event compute_finished = (*m_program_naive_sqdiff)(
					m_kernel_naive_sqdiff_no_mask,
					pre_compute_events.begin(), pre_compute_events.end(),
					exec_params,
					input_image,
					kernel_image,
					*m_output_buffer,
					cl_int2{static_cast<int>(input_image_desc.dimensions.width), static_cast<int>(input_image_desc.dimensions.height)},
					cl_int2{static_cast<int>(kernel_image_desc.dimensions.width), static_cast<int>(kernel_image_desc.dimensions.height)},
					cl_float2{std::sinf(static_cast<float>(texture_rotation)), std::cosf(static_cast<float>(texture_rotation))}
				);
				// read result
				pre_compute_events.clear();
				pre_compute_events.push_back(std::move(compute_finished));
				read_output_image(match_res_out.total_cost_matrix, out_dims, pre_compute_events).wait();
			}
			
			inline void ocl_template_matching::matching_policies::impl::CLMatcherImpl::find_best_matches(MatchingResult& match_res_out)
			{
				// TODO: Implement parallel extraction of best match
			}

			inline cv::Vec3i ocl_template_matching::matching_policies::impl::CLMatcherImpl::response_dimensions(
				const Texture& texture,
				const Texture& kernel,
				double texture_rotation) const
			{
				return get_response_dimensions(texture, kernel);
			}

			inline match_response_cv_mat_t ocl_template_matching::matching_policies::impl::CLMatcherImpl::response_image_data_type(
				const Texture& texture,
				const Texture& kernel,
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

void ocl_template_matching::matching_policies::CLMatcher::compute_response(
	const Texture& texture,
	const cv::Mat& texture_mask,
	const Texture& kernel,
	double texture_rotation,
	MatchingResult& match_res_out)
{
	impl()->compute_response(texture, texture_mask, kernel, texture_rotation, match_res_out);
}

void ocl_template_matching::matching_policies::CLMatcher::compute_response(
	const Texture& texture,
	const Texture& kernel,
	const cv::Mat& kernel_mask,
	double texture_rotation,
	MatchingResult& match_res_out)
{
	impl()->compute_response(texture, kernel, kernel_mask, texture_rotation, match_res_out);
}

void ocl_template_matching::matching_policies::CLMatcher::compute_response(
	const Texture& texture,
	const Texture& kernel,
	double texture_rotation,
	MatchingResult& match_res_out)
{
	impl()->compute_response(texture, kernel, texture_rotation, match_res_out);
}

void ocl_template_matching::matching_policies::CLMatcher::find_best_matches(MatchingResult& match_res_out)
{
	impl()->find_best_matches(match_res_out);
}

cv::Vec3i ocl_template_matching::matching_policies::CLMatcher::response_dimensions(
	const Texture& texture,
	const Texture& kernel,
	double texture_rotation) const
{
	return impl()->response_dimensions(texture, kernel, texture_rotation);
}

ocl_template_matching::match_response_cv_mat_t ocl_template_matching::matching_policies::CLMatcher::response_image_data_type(
	const Texture& texture,
	const Texture& kernel,
	double texture_rotation) const
{
	return impl()->response_image_data_type(texture, kernel, texture_rotation);
}
#pragma endregion