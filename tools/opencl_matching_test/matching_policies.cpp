#include <matching_policies.hpp>
#include <simple_cl.hpp>
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <vector>
#include <stack>

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
					const Texture& kernel,
					const cv::Mat& kernel_mask,
					double texture_rotation,
					MatchingResult& match_res_out
				);

				void find_best_matches(MatchingResult& match_res_out);
				void find_best_matches(MatchingResult& match_res_out, const cv::Mat& texture_mask);

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
				static cv::Vec2d get_cv_image_normalizer(const cv::Mat& img);
				
				void prepare_input_image(const Texture& input, std::vector<simple_cl::cl::Event>& event_list, bool invalidate = false, bool blocking = true);
				void prepare_kernel_image(const Texture& kernel_texture);
				void prepare_texture_mask(const cv::Mat& texture_mask);
				void prepare_kernel_mask(const cv::Mat& kernel_mask);
				void prepare_output_image(const Texture& input, const Texture& kernel);
				simple_cl::cl::Event clear_output_image(float value = std::numeric_limits<float>::max());

				// resource handling
				void invalidate_input_texture(const std::string& texid);

				// get results
				simple_cl::cl::Event read_output_image(cv::Mat& out_mat, const cv::Vec3i& output_size, const std::vector<simple_cl::cl::Event>& wait_for);

				// ------------------------------------------------------------ data members ----------------------------------------------------------------
				struct InputTextureData
				{
					std::vector<cv::Mat> data;
					std::size_t width;
					std::size_t height;
					std::size_t num_channels;
				};

				struct InputImage
				{
					std::vector<std::unique_ptr<simple_cl::cl::Image>> images;
				};

				ocl_template_matching::matching_policies::CLMatcher::DeviceSelectionPolicy m_selection_policy;
				std::size_t m_max_tex_cache_size;

				// TODO: Think about texture cache!

				// Output buffer. Only use a single output image and enlarge it when necessary.
				std::unique_ptr<simple_cl::cl::Image> m_output_buffer_a;
				// Second output buffer in case we have more than 4 feature maps. This is used to ping-pong the result between batches
				// of four feature maps, accumulating the total error.
				std::unique_ptr<simple_cl::cl::Image> m_output_buffer_b;

				// input textures
				// used to manage indices of textures which became invalid. Instead of deleting from the vectors, create new resources at the free indices.
				std::stack<std::size_t, std::vector<std::size_t>> m_free_indices;
				// used to cache converted floating point data from input textures
				std::vector<InputTextureData> m_texture_cache;
				// vector of collection of opencl images. Each image can hold 4 feature maps.
				std::vector<InputImage> m_input_images;
				// texture mask. This needs to be updated on every match.
				std::unique_ptr<simple_cl::cl::Image> m_texture_mask;
				// kernel images. Again, 4 feature maps per image. Needs to be updated on every match.
				std::vector<std::unique_ptr<simple_cl::cl::Image>> m_kernel_images;
				// maps texture id to index into texture cache and collection of input images
				std::unordered_map<std::string, std::size_t> m_texture_index_map;

				// OpenCL context
				std::shared_ptr<simple_cl::cl::Context> m_cl_context;

				// OpenCL programs
				std::unique_ptr<simple_cl::cl::Program> m_program_naive_sqdiff;

				// Kernel handles
				simple_cl::cl::Program::CLKernelHandle m_kernel_naive_sqdiff;
				simple_cl::cl::Program::CLKernelHandle m_kernel_naive_sqdiff_masked;
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
					simple_cl::cl::Image::ImageType::Image2D,	// One array slice per response channel
					simple_cl::cl::Image::ImageDimensions{
						static_cast<std::size_t>(input_tex.response.cols()),				// width
						static_cast<std::size_t>(input_tex.response.rows()),				// height
						static_cast<std::size_t>(input_tex.response.num_channels())			// number of slices
					},
					simple_cl::cl::Image::ImageChannelOrder::RGBA,		// One red channel per slice
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
					simple_cl::cl::Image::ImageType::Image2D,	// One array slice per response channel
					simple_cl::cl::Image::ImageDimensions{
						static_cast<std::size_t>(kernel_tex.response.cols()),				// width
						static_cast<std::size_t>(kernel_tex.response.rows()),				// height
						static_cast<std::size_t>(kernel_tex.response.num_channels())			// number of slices
					},
					simple_cl::cl::Image::ImageChannelOrder::RGBA,		// One red channel per slice
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

			inline cv::Vec2d ocl_template_matching::matching_policies::impl::CLMatcherImpl::get_cv_image_normalizer(const cv::Mat& img)
			{
				// normalize signed integer values to [-1, 1] and unsigned values to [0, 1]
				switch(img.depth())
				{
					case CV_8U:
						return cv::Vec2d(1.0 / 255.0, 0.0);
						break;
					case CV_8S:
						return cv::Vec2d(2.0 / (127.0 - (-128.0)), ((-2.0 * -128.0) / (127.0 - (-128.0))) + 1.0);
						break;
					case CV_16U:
						return cv::Vec2d(1.0 / 65535.0, 0.0);
						break;
					case CV_16S:
						return cv::Vec2d(2.0 / (32767.0 - (-32768.0)), ((-2.0 * -32768.0) / (32767.0 - (-32768.0))) + 1.0);
						break;
					case CV_32S:
						return cv::Vec2d(2.0 / (2147483647.0 - (-2147483648.0)), ((-2.0 * -2147483648.0) / (2147483647.0 - (-2147483648.0))) + 1.0);
						break;
					case CV_32F:
						return cv::Vec2d(1.0, 0.0);
						break;
					case CV_64F:
						return cv::Vec2d(1.0, 0.0);
						break;
					default:
						return cv::Vec2d(1.0, 0.0);
						break;
				}
			}

			// ctors and stuff

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

			inline void ocl_template_matching::matching_policies::impl::CLMatcherImpl::invalidate_input_texture(const std::string& texid)
			{
				std::size_t index{m_texture_index_map.at(texid)};
				m_texture_index_map.erase(texid);
				m_free_indices.push(index);
			}

			inline void ocl_template_matching::matching_policies::impl::CLMatcherImpl::initialize_opencl_state(const std::shared_ptr<simple_cl::cl::Context>& clcontext)
			{
				// save context
				m_cl_context = clcontext;
				// create and compile programs
				m_program_naive_sqdiff.reset(new simple_cl::cl::Program(kernels::sqdiff_naive_src, kernels::sqdiff_naive_copt, m_cl_context));
				// retrieve kernel handles
				m_kernel_naive_sqdiff = m_program_naive_sqdiff->getKernel("sqdiff_naive");
				m_kernel_naive_sqdiff_masked = m_program_naive_sqdiff->getKernel("sqdiff_naive_masked");
			}

			inline void ocl_template_matching::matching_policies::impl::CLMatcherImpl::cleanup_opencl_state()
			{
			}

			inline void ocl_template_matching::matching_policies::impl::CLMatcherImpl::prepare_input_image(const Texture& input, std::vector<simple_cl::cl::Event>& event_list, bool invalidate = false, bool blocking = true)
			{
				// use async api calls where possible to reduce gpu bubbles. WHY TF DOES OPENCV NOT HAVE MOVE CONTRUCTORS???
				static std::vector<simple_cl::cl::Event> events;
				events.clear();
				// for writing images
				simple_cl::cl::Image::HostFormat host_fmt{
					simple_cl::cl::Image::HostChannelOrder{
						4ull,
						{
							simple_cl::cl::Image::ColorChannel::R,
							simple_cl::cl::Image::ColorChannel::G,
							simple_cl::cl::Image::ColorChannel::B,
							simple_cl::cl::Image::ColorChannel::A
						}
					},
					simple_cl::cl::Image::HostDataType::FLOAT,
					simple_cl::cl::Image::HostPitch{}
				};
				simple_cl::cl::Image::ImageRegion img_region{
					simple_cl::cl::Image::ImageOffset{0ull, 0ull, 0ull},
					simple_cl::cl::Image::ImageDimensions{static_cast<std::size_t>(input.response.cols()), static_cast<std::size_t>(input.response.cols()), 1ull}
				};
				// one input image per 4 feature maps!
				std::size_t num_feature_maps{static_cast<std::size_t>(input.response.num_channels())};
				std::size_t num_images{num_feature_maps / 4ull + (num_feature_maps % 4ull != 0ull ? 1ull : 0ull)};
				// normalizer
				auto normalizer{get_cv_image_normalizer(input.response[0])};
				// opencl image desc
				auto desc = make_input_image_desc(input);
				// if texture aready exists in the cache, reuse it!
				if(m_texture_index_map.count(input.id))
				{
					std::size_t tex_index{m_texture_index_map[input.id]};
					InputTextureData& texture{m_texture_cache[tex_index]};
					InputImage& image{m_input_images[tex_index]};
					// Size equal to size of existing texture ?
					bool size_matches{(texture.width == static_cast<std::size_t>(input.response.cols()) && texture.height == static_cast<std::size_t>(input.response.rows()) && texture.num_channels == num_feature_maps)};
					if(size_matches && !invalidate)
					{
						return;
					}
					else
					{
						// convert texture to float data
						texture.data.clear();
						// single channels
						cv::Mat float_channels[4];
						for(std::size_t i{0ull}; i < num_images; ++i)
						{
							for(std::size_t c{0ull}; c < 4ull; ++c)
							{
								std::size_t channel_idx{i * 4ull + c};
								if(channel_idx < num_feature_maps)
								{
									input.response[channel_idx].convertTo(float_channels[c], CV_32FC1, normalizer[0], normalizer[1]);
								}
								else
								{
									float_channels[c] = cv::Mat(input.response[channel_idx].rows, input.response[channel_idx].cols, CV_32FC1);
									float_channels[c] = cv::Scalar(0.0);
								}
							}
							cv::Mat rgba_img;
							cv::merge(&float_channels[0], 4ull, rgba_img);
							texture.data[i] = std::move(rgba_img);
						}
						// If size matches we can simply upload the new data.
						if(size_matches) 
						{
							for(std::size_t i{0ull}; i < num_images; ++i)
							{
								events.push_back(image.images[i]->write(img_region, host_fmt, texture.data[i].data, false));
							}
							// wait until the upload is finished
							if(blocking)
								simple_cl::cl::wait_for_events(events.begin(), events.end());
							else
								event_list.insert(event_list.end(), events.begin(), events.end());
						}
						// if size of new texture is smaller, reuse the existing memory instead of creating a new cl image
						else if((image.images[0]->width() >= static_cast<std::size_t>(input.response.cols())) && (image.images[0]->height() >= static_cast<std::size_t>(input.response.rows())) && (image.images.size() * 4ull >= num_feature_maps))
						{
							// store new sizes
							texture.width = static_cast<std::size_t>(input.response.cols());
							texture.height = static_cast<std::size_t>(input.response.rows());
							texture.num_channels = num_feature_maps;

							for(std::size_t i{0ull}; i < num_images; ++i)
							{
								events.push_back(image.images[i]->write(img_region, host_fmt, texture.data[i].data, false));
							}							
							// wait until the upload is finished
							if(blocking)
								simple_cl::cl::wait_for_events(events.begin(), events.end());
							else
								event_list.insert(event_list.end(), events.begin(), events.end());
						}
						// old image is too small. We have to create new images.
						else
						{
							// store new sizes
							texture.width = static_cast<std::size_t>(input.response.cols());
							texture.height = static_cast<std::size_t>(input.response.rows());
							texture.num_channels = num_feature_maps;

							// clear old images
							image.images.clear();
							for(std::size_t i{0ull}; i < num_images; ++i)
							{
								// create new image
								image.images.push_back(std::unique_ptr<simple_cl::cl::Image>(new simple_cl::cl::Image(m_cl_context, desc)));
								// write new data
								events.push_back(image.images.back()->write(img_region, host_fmt, texture.data[i].data, false));
							}
							// wait until the upload is finished
							if(blocking)
								simple_cl::cl::wait_for_events(events.begin(), events.end());
							else
								event_list.insert(event_list.end(), events.begin(), events.end());
						}
					}
				}
				else
				{
					// convert texture to float data
					InputTextureData texture_data;
					texture_data.width = static_cast<std::size_t>(input.response.cols());
					texture_data.height = static_cast<std::size_t>(input.response.rows());
					texture_data.num_channels = num_feature_maps;
					texture_data.data.reserve(num_images);

					// opencl images
					InputImage input_image;
					input_image.images.reserve(num_images);

					// single channels
					cv::Mat float_channels[4];

					// convert input data
					for(std::size_t i{0ull}; i < num_images; ++i)
					{
						for(std::size_t c{0ull}; c < 4ull; ++c)
						{
							std::size_t channel_idx{i * 4ull + c};
							if(channel_idx < num_feature_maps)
							{
								input.response[channel_idx].convertTo(float_channels[c], CV_32FC1, normalizer[0], normalizer[1]);
							}
							else
							{
								float_channels[c] = cv::Mat(input.response[channel_idx].rows, input.response[channel_idx].cols, CV_32FC1);
								float_channels[c] = cv::Scalar(0.0);
							}
						}
						cv::Mat rgba_img;
						cv::merge(&float_channels[0], 4ull, rgba_img);
						texture_data.data.push_back(std::move(rgba_img));
					}

					// create images and write image data
					for(std::size_t i{0ull}; i < num_images; ++i)
					{
						// create new image
						input_image.images.push_back(std::unique_ptr<simple_cl::cl::Image>(new simple_cl::cl::Image(m_cl_context, desc)));
						// write new data
						events.push_back(input_image.images.back()->write(img_region, host_fmt, texture_data.data[i].data, false));
					}

					// are there free indices?
					if(m_free_indices.empty())
					{
						// add float data to texture cache
						m_texture_cache.push_back(std::move(texture_data));
						// add cl image to input images
						m_input_images.push_back(std::move(input_image));
						// get texture index
						std::size_t texture_index{m_texture_cache.size() - 1};
						// add index to index map
						m_texture_index_map[input.id] = texture_index;
					}
					else // reuse free slot
					{
						// get next free index
						std::size_t texture_index{m_free_indices.top()};
						m_texture_cache[texture_index] = std::move(texture_data);
						m_input_images[texture_index] = std::move(input_image);
						// store new index in the map
						m_texture_index_map[input.id] = texture_index;
					}

					// wait until the upload is finished
					if(blocking)
						simple_cl::cl::wait_for_events(events.begin(), events.end());
					else
						event_list.insert(event_list.end(), events.begin(), events.end());
				}
			}

			inline void ocl_template_matching::matching_policies::impl::CLMatcherImpl::prepare_kernel_image(const Texture& kernel_texture)
			{
			}

			inline void ocl_template_matching::matching_policies::impl::CLMatcherImpl::prepare_texture_mask(const cv::Mat& texture_mask)
			{
			}

			inline void ocl_template_matching::matching_policies::impl::CLMatcherImpl::prepare_kernel_mask(const cv::Mat& kernel_mask)
			{
			}

			void ocl_template_matching::matching_policies::impl::CLMatcherImpl::prepare_output_image(const Texture& input, const Texture& kernel)
			{
				auto dims{get_response_dimensions(input, kernel)};
				if(m_output_buffer_a) // if output image already exists
				{
					// recreate output image only if it is too small for the new input - kernel combination
					if(static_cast<std::size_t>(dims[0]) > m_output_buffer_a->width() || static_cast<std::size_t>(dims[1]) > m_output_buffer_a->height() || static_cast<std::size_t>(dims[2]) > m_output_buffer_a->layers())
					{
						auto output_desc{make_output_image_desc(input, kernel)};
						m_output_buffer_a.reset(new simple_cl::cl::Image(m_cl_context, output_desc));
					}
				}
				else
				{
					auto output_desc{make_output_image_desc(input, kernel)};
					m_output_buffer_a.reset(new simple_cl::cl::Image(m_cl_context, output_desc));
				}
			}

			simple_cl::cl::Event CLMatcherImpl::clear_output_image(float value)
			{
				simple_cl::cl::Image::ImageRegion region{
					simple_cl::cl::Image::ImageOffset{0ull, 0ull, 0ull},
					simple_cl::cl::Image::ImageDimensions{m_output_buffer_a->width(), m_output_buffer_a->height(), m_output_buffer_a->layers()}
				};
				return m_output_buffer_a->fill(simple_cl::cl::Image::FillColor{value}, region);
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
				return m_output_buffer_a->read(region, hostfmt, out_mat.data, wait_for.begin(), wait_for.end());
			}

			inline void ocl_template_matching::matching_policies::impl::CLMatcherImpl::compute_response(
				const Texture& texture,
				const Texture& kernel,
				const cv::Mat& kernel_mask,
				double texture_rotation,
				MatchingResult& match_res_out)
			{
				
			}

			inline void ocl_template_matching::matching_policies::impl::CLMatcherImpl::compute_response(
				const Texture& texture,
				const Texture& kernel,
				double texture_rotation,
				MatchingResult& match_res_out)
			{
				static std::vector<simple_cl::cl::Event> pre_compute_events;
				pre_compute_events.clear();
				
			}
			
			inline void ocl_template_matching::matching_policies::impl::CLMatcherImpl::find_best_matches(MatchingResult& match_res_out)
			{
				// TODO: Implement parallel extraction of best match
			}

			inline void ocl_template_matching::matching_policies::impl::CLMatcherImpl::find_best_matches(MatchingResult& match_res_out, const cv::Mat& texture_mask)
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

void ocl_template_matching::matching_policies::CLMatcher::find_best_matches(MatchingResult& match_res_out, const cv::Mat& texture_mask)
{
	impl()->find_best_matches(match_res_out, texture_mask);
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