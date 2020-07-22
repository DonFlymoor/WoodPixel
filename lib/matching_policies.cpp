#include <matching_policies.hpp>
#include <simple_cl.hpp>
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <vector>
#include <stack>

namespace ocl_patch_matching
{
	namespace matching_policies
	{
		namespace impl
		{
			#pragma region CLMatcherImpl declaration
			// include the kernel files
			#include <kernels/kernel_sqdiff_naive.hpp>
			#include <kernels/kernel_sqdiff_constant.hpp>
			#include <kernels/kernel_sqdiff_constant_local.hpp>
			#include <kernels/kernel_sqdiff_constant_local_masked.hpp>
			#include <kernels/kernel_erode_masked.hpp>
			#include <kernels/kernel_erode.hpp>
			#include <kernels/kernel_erode_masked_local.hpp>
			#include <kernels/kernel_erode_local.hpp>
			#include <kernels/find_min.hpp>

			class CLMatcherImpl
			{
			public:
				CLMatcherImpl(
					ocl_patch_matching::matching_policies::CLMatcher::DeviceSelectionPolicy device_selection_policy, 
					std::size_t max_texture_cache_memory,
					std::size_t local_block_size,
					std::size_t constant_kernel_max_pixels,
					std::size_t local_buffer_max_pixels, 
					ocl_patch_matching::matching_policies::CLMatcher::ResultOrigin result_origin,
					bool use_local_buffer_for_matching,
					bool use_local_buffer_for_erode
				);
				~CLMatcherImpl() noexcept;
				CLMatcherImpl(const CLMatcherImpl&) = delete;
				CLMatcherImpl(CLMatcherImpl&&) = delete;
				CLMatcherImpl& operator=(const CLMatcherImpl&) = delete;
				CLMatcherImpl& operator=(CLMatcherImpl&&) = delete;

				std::size_t platform_id() const;
				std::size_t device_id() const;

				void initialize_opencl_state(const std::shared_ptr<simple_cl::cl::Context>& clcontext);
				void cleanup_opencl_state();

				void compute_matches(
					const Texture& texture,
					const Texture& kernel,
					double texture_rotation,
					MatchingResult& match_res_out
				);

				void compute_matches(
					const Texture& texture,
					const Texture& kernel,
					const cv::Mat& kernel_mask,
					double texture_rotation,
					MatchingResult& match_res_out
				);

				void compute_matches(
					const Texture& texture,
					const cv::Mat& texture_mask,
					const Texture& kernel,
					double texture_rotation,
					MatchingResult& match_res_out,
					bool erode_texture_mask
				);

				void compute_matches(
					const Texture& texture,
					const cv::Mat& texture_mask,
					const Texture& kernel,
					const cv::Mat& kernel_mask,
					double texture_rotation,
					MatchingResult& match_res_out,
					bool erode_texture_mask
				);

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
				static simple_cl::cl::Image::ImageDesc make_output_image_desc(const Texture& input_tex, const Texture& kernel_tex, double texture_rotation, const cv::Size& response_dims);
				static simple_cl::cl::Image::ImageDesc make_kernel_image_desc(const Texture& kernel_tex);
				static simple_cl::cl::Image::ImageDesc make_mask_image_desc(const cv::Mat& texture_mask);
				static simple_cl::cl::Image::ImageDesc make_mask_output_image_desc(const cv::Mat& texture_mask);
				static simple_cl::cl::Image::ImageDesc make_kernel_mask_image_desc(const cv::Mat& kernel_mask);

				static cv::Size get_response_dimensions(const Texture& texture, const Texture& kernel, double texture_rotation, const cv::Point& kernel_anchor);
				static cv::Vec2d get_cv_image_normalizer(const cv::Mat& img);
				
				void prepare_input_image(const Texture& input, std::vector<simple_cl::cl::Event>& event_list, bool invalidate = false, bool blocking = true);				
				void prepare_texture_mask(const cv::Mat& texture_mask, std::vector<simple_cl::cl::Event>& event_list, bool blocking = true);

				void prepare_kernel_image(const Texture& kernel_texture, std::vector<simple_cl::cl::Event>& event_list, bool blocking = true);
				void prepare_kernel_mask(const cv::Mat& kernel_mask, std::vector<simple_cl::cl::Event>& event_list, bool blocking = true);
				void prepare_kernel_buffer(const Texture& kernel_texture, std::vector<simple_cl::cl::Event>& event_list, bool blocking = true);
				void prepare_kernel_mask_buffer(const cv::Mat& kernel_mask, std::vector<simple_cl::cl::Event>& event_list, bool blocking = true);

				void prepare_output_image(const Texture& input, const Texture& kernel, double texture_rotation, const cv::Size& response_dims);
				void prepare_erode_output_image(const cv::Mat& texture_mask);
				simple_cl::cl::Event clear_output_image_a(float value = 0.0f);
				simple_cl::cl::Event clear_output_image_b(float value = 0.0f);

				void prepare_find_min_output_buffer(const cv::Size& out_size, std::size_t local_work_size_xy, std::size_t& global_work_size_x, std::size_t& global_work_size_y, std::size_t& local_buffer_size);

				// resource handling
				void invalidate_input_texture(const std::string& texid);

				// decide if we should use constant memory kernel or images
				bool use_constant_kernel(const Texture& kernel, const cv::Mat& kernel_mask) const;
				bool use_constant_kernel(const Texture& kernel) const;
				bool use_constant_kernel(const cv::Mat& kernel_mask) const;

				// decide when to use local memory optimization
				bool use_local_mem(const cv::Vec4i& kernel_overlaps, std::size_t used_local_mem, std::size_t local_work_size, std::size_t max_pixels, std::size_t size_per_pixel);

				// decide which work group size to use, the passed local_block_size parameter is the maximum
				std::size_t get_local_work_size(const simple_cl::cl::Program::CLKernelHandle& kernel) const;

				// calculate rotated kernel bounding box and padding sizes
				static void calculate_rotated_kernel_dims(cv::Size& rotated_kernel_size, cv::Vec4i& rotated_kernel_overlaps, const Texture& kernel, double texture_rotation, const cv::Point& anchor = cv::Point{-1, -1});

				// get results
				simple_cl::cl::Event read_output_image(cv::Mat& out_mat, const cv::Size& output_size, const std::vector<simple_cl::cl::Event>& wait_for, bool out_a);
				simple_cl::cl::Event read_eroded_texture_mask_image(cv::Mat& out_mat, const cv::Size& output_size, const std::vector<simple_cl::cl::Event>& wait_for);
				void read_min_pos_and_cost(MatchingResult& res, const std::vector<simple_cl::cl::Event>& wait_for, const cv::Point& res_coord_offset);

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
					InputTextureData data;
				};

				struct KernelImage
				{
					std::vector<std::unique_ptr<simple_cl::cl::Image>> images;
					std::size_t num_channels;
				};

				struct KernelBuffer
				{
					std::unique_ptr<simple_cl::cl::Buffer> buffer;
				};

				struct KernelMaskBuffer
				{
					std::unique_ptr<simple_cl::cl::Buffer> buffer;
				};

				struct FindMinBuffer
				{
					std::unique_ptr<simple_cl::cl::Buffer> buffer;
					std::size_t num_work_groups[2];
				};

				// specifies result orgin. either upper left corner or center
				ocl_patch_matching::matching_policies::CLMatcher::ResultOrigin m_result_origin;
				// decide which opencl device to select if there are more than one
				ocl_patch_matching::matching_policies::CLMatcher::DeviceSelectionPolicy m_selection_policy;
				// enable / disable local buffer for matching step
				bool m_use_local_buffer_for_matching;
				// enable / disable local buffer for erode step
				bool m_use_local_buffer_for_erode;
				// ignored for now
				std::size_t m_max_tex_cache_size;
				// size of local work groups (square blocks)
				std::size_t m_local_block_size;
				// maximum number of pixels of a kernel for which we use constant memory
				std::size_t m_constant_kernel_max_pixels;
				// maximum number of pixels of the workgroup + kernel overlap region for which we use local memory buffers
				std::size_t m_local_buffer_max_pixels;


				// Output buffer. Only use a single output image and enlarge it when necessary.
				std::unique_ptr<simple_cl::cl::Image> m_output_buffer_a;
				// Second output buffer in case we have more than 4 feature maps. This is used to ping-pong the result between batches
				// of four feature maps, accumulating the total error.
				std::unique_ptr<simple_cl::cl::Image> m_output_buffer_b;

				// output buffer for erode
				std::unique_ptr<simple_cl::cl::Image> m_output_texture_mask_eroded;

				// output buffer for find_min
				FindMinBuffer m_output_buffer_find_min;
				// texture mask. This needs to be updated on every match.
				std::unique_ptr<simple_cl::cl::Image> m_texture_mask;

				// input textures
				// used to manage indices of textures which became invalid. Instead of deleting from the vectors, create new resources at the free indices.
				std::stack<std::size_t, std::vector<std::size_t>> m_free_indices;
				// vector of collection of opencl images. Each image can hold 4 feature maps.
				std::vector<InputImage> m_input_images;
				// maps texture id to index into texture cache and collection of input images
				std::unordered_map<std::string, std::size_t> m_texture_index_map;

				// kernel
				// kernel images. Again, 4 feature maps per image. Needs to be updated on every match.
				KernelImage m_kernel_image;
				// if the kernel fits into constant memory, we can use buffers instead
				KernelBuffer m_kernel_buffer;
				// kernel mask
				std::unique_ptr<simple_cl::cl::Image> m_kernel_mask;
				// if this kernel fits into constant memory, we can use buffers instead
				KernelMaskBuffer m_kernel_mask_buffer;			
				
				// OpenCL context
				std::shared_ptr<simple_cl::cl::Context> m_cl_context;

				// OpenCL programs
				std::unique_ptr<simple_cl::cl::Program> m_program_naive_sqdiff;
				std::unique_ptr<simple_cl::cl::Program> m_program_sqdiff_constant;
				std::unique_ptr<simple_cl::cl::Program> m_program_sqdiff_constant_local;
				std::unique_ptr<simple_cl::cl::Program> m_program_sqdiff_constant_local_masked;
				std::unique_ptr<simple_cl::cl::Program> m_program_erode_masked;
				std::unique_ptr<simple_cl::cl::Program> m_program_erode;
				std::unique_ptr<simple_cl::cl::Program> m_program_erode_masked_local;
				std::unique_ptr<simple_cl::cl::Program> m_program_erode_local;
				std::unique_ptr<simple_cl::cl::Program> m_program_find_min;

				// Kernel handles
				simple_cl::cl::Program::CLKernelHandle m_kernel_naive_sqdiff;	
				simple_cl::cl::Program::CLKernelHandle m_kernel_naive_sqdiff_nth_pass;
				simple_cl::cl::Program::CLKernelHandle m_kernel_naive_sqdiff_masked;
				simple_cl::cl::Program::CLKernelHandle m_kernel_naive_sqdiff_masked_nth_pass;

				simple_cl::cl::Program::CLKernelHandle m_kernel_constant_sqdiff;
				simple_cl::cl::Program::CLKernelHandle m_kernel_constant_sqdiff_nth_pass;
				simple_cl::cl::Program::CLKernelHandle m_kernel_constant_sqdiff_masked;
				simple_cl::cl::Program::CLKernelHandle m_kernel_constant_sqdiff_masked_nth_pass;

				simple_cl::cl::Program::CLKernelHandle m_kernel_constant_sqdiff_local;
				simple_cl::cl::Program::CLKernelHandle m_kernel_constant_sqdiff_local_nth_pass;
				simple_cl::cl::Program::CLKernelHandle m_kernel_constant_sqdiff_local_masked;
				simple_cl::cl::Program::CLKernelHandle m_kernel_constant_sqdiff_local_masked_nth_pass;

				simple_cl::cl::Program::CLKernelHandle m_kernel_erode_masked;
				simple_cl::cl::Program::CLKernelHandle m_kernel_erode_constant_masked;
				simple_cl::cl::Program::CLKernelHandle m_kernel_erode_masked_local;

				simple_cl::cl::Program::CLKernelHandle m_kernel_erode;
				simple_cl::cl::Program::CLKernelHandle m_kernel_erode_local;

				simple_cl::cl::Program::CLKernelHandle m_kernel_find_min;
				simple_cl::cl::Program::CLKernelHandle m_kernel_find_min_masked;
			};
			#pragma endregion

			#pragma region CLMatcherImpl implementation

			// static functions
			inline cv::Size ocl_patch_matching::matching_policies::impl::CLMatcherImpl::get_response_dimensions(const Texture& texture, const Texture& kernel, double texture_rotation, const cv::Point& kernel_anchor)
			{
				cv::Size rotated_kernel_size;
				cv::Vec4i rotated_kernel_overlaps;
				calculate_rotated_kernel_dims(rotated_kernel_size, rotated_kernel_overlaps, kernel, texture_rotation, kernel_anchor);

				return cv::Size{
					texture.response.cols() - rotated_kernel_overlaps[0] - rotated_kernel_overlaps[1],
					texture.response.rows() - rotated_kernel_overlaps[2] - rotated_kernel_overlaps[3]
				};
			}

			inline simple_cl::cl::Image::ImageDesc ocl_patch_matching::matching_policies::impl::CLMatcherImpl::make_input_image_desc(const Texture& input_tex)
			{
  				return simple_cl::cl::Image::ImageDesc{
					simple_cl::cl::Image::ImageType::Image2D,	// One array slice per response channel
					simple_cl::cl::Image::ImageDimensions{
						static_cast<std::size_t>(input_tex.response.cols()),				// width
						static_cast<std::size_t>(input_tex.response.rows()),				// height
						1ull																// number of slices
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

			inline simple_cl::cl::Image::ImageDesc ocl_patch_matching::matching_policies::impl::CLMatcherImpl::make_output_image_desc(const Texture& input_tex, const Texture& kernel_tex, double texture_rotation, const cv::Size& response_dims)
			{
				return simple_cl::cl::Image::ImageDesc{
					simple_cl::cl::Image::ImageType::Image2D,		// One response channel
					simple_cl::cl::Image::ImageDimensions{
						static_cast<std::size_t>(response_dims.width),				// width
						static_cast<std::size_t>(response_dims.height),				// height
						1ull													// number of slices
					},
					simple_cl::cl::Image::ImageChannelOrder::R,		// One red channel
					simple_cl::cl::Image::ImageChannelType::FLOAT,	// Single precision floating point data
					simple_cl::cl::MemoryFlags{
						simple_cl::cl::DeviceAccess::ReadWrite,		// Kernel may read and write
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

			inline simple_cl::cl::Image::ImageDesc ocl_patch_matching::matching_policies::impl::CLMatcherImpl::make_kernel_image_desc(const Texture& kernel_tex)
			{
				return simple_cl::cl::Image::ImageDesc{
					simple_cl::cl::Image::ImageType::Image2D,	// One array slice per response channel
					simple_cl::cl::Image::ImageDimensions{
						static_cast<std::size_t>(kernel_tex.response.cols()),				// width
						static_cast<std::size_t>(kernel_tex.response.rows()),				// height
						1ull																// number of slices
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

			inline simple_cl::cl::Image::ImageDesc ocl_patch_matching::matching_policies::impl::CLMatcherImpl::make_mask_image_desc(const cv::Mat& texture_mask)
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

			inline simple_cl::cl::Image::ImageDesc ocl_patch_matching::matching_policies::impl::CLMatcherImpl::make_mask_output_image_desc(const cv::Mat& texture_mask)
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
						simple_cl::cl::DeviceAccess::ReadWrite,		// Kernel may only write
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
			
			inline simple_cl::cl::Image::ImageDesc ocl_patch_matching::matching_policies::impl::CLMatcherImpl::make_kernel_mask_image_desc(const cv::Mat& kernel_mask)
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

			inline cv::Vec2d ocl_patch_matching::matching_policies::impl::CLMatcherImpl::get_cv_image_normalizer(const cv::Mat& img)
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

			inline ocl_patch_matching::matching_policies::impl::CLMatcherImpl::CLMatcherImpl(
				ocl_patch_matching::matching_policies::CLMatcher::DeviceSelectionPolicy device_selection_policy,
				std::size_t max_texture_cache_memory,
				std::size_t local_block_size,
				std::size_t constant_kernel_maxdim,
				std::size_t local_buffer_max_pixels,
				ocl_patch_matching::matching_policies::CLMatcher::ResultOrigin result_origin,
				bool use_local_buffer_for_matching,
				bool use_local_buffer_for_erode) :
					m_selection_policy(device_selection_policy),
					m_max_tex_cache_size(max_texture_cache_memory),
					m_kernel_image{std::vector<std::unique_ptr<simple_cl::cl::Image>>(), 0ull},
					m_local_block_size{local_block_size},
					m_constant_kernel_max_pixels{constant_kernel_maxdim},
					m_local_buffer_max_pixels{local_buffer_max_pixels},
					m_result_origin{result_origin},
					m_use_local_buffer_for_matching{use_local_buffer_for_matching},
					m_use_local_buffer_for_erode{use_local_buffer_for_erode}
			{
				if(!simple_cl::util::is_power_of_two(local_block_size) || local_block_size == 0ull)
					throw std::invalid_argument("local_block_size must be a positive power of two.");
			}

			inline ocl_patch_matching::matching_policies::impl::CLMatcherImpl::~CLMatcherImpl() noexcept
			{
			}

			inline void ocl_patch_matching::matching_policies::impl::CLMatcherImpl::select_platform_and_device(std::size_t& platform_idx, std::size_t& device_idx) const
			{
				auto pdevinfo = simple_cl::cl::Context::read_platform_and_device_info();
				std::size_t plat_idx{0ull};
				std::size_t dev_idx{0ull};

				if(m_selection_policy == ocl_patch_matching::matching_policies::CLMatcher::DeviceSelectionPolicy::FirstSuitableDevice)
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
							case ocl_patch_matching::matching_policies::CLMatcher::DeviceSelectionPolicy::MostComputeUnits:
								if(pdevinfo[p].devices[d].max_compute_units > pdevinfo[plat_idx].devices[dev_idx].max_compute_units) 
								{ 
									plat_idx = p;
									dev_idx = d;
								}
								break;
							case ocl_patch_matching::matching_policies::CLMatcher::DeviceSelectionPolicy::MostGPUThreads:
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

			inline std::size_t ocl_patch_matching::matching_policies::impl::CLMatcherImpl::platform_id() const
			{
				std::size_t pidx, didx;
				select_platform_and_device(pidx, didx);
				return pidx;
			}

			inline std::size_t ocl_patch_matching::matching_policies::impl::CLMatcherImpl::device_id() const
			{
				std::size_t pidx, didx;
				select_platform_and_device(pidx, didx);
				return didx;
			}

			inline void ocl_patch_matching::matching_policies::impl::CLMatcherImpl::invalidate_input_texture(const std::string& texid)
			{
				std::size_t index{m_texture_index_map.at(texid)};
				m_texture_index_map.erase(texid);
				m_free_indices.push(index);
			}

			inline void ocl_patch_matching::matching_policies::impl::CLMatcherImpl::initialize_opencl_state(const std::shared_ptr<simple_cl::cl::Context>& clcontext)
			{
				// save context
				m_cl_context = clcontext;
				// create and compile programs
				m_program_naive_sqdiff.reset(new simple_cl::cl::Program(kernels::sqdiff_naive_src, kernels::sqdiff_naive_copt, m_cl_context));
				m_program_sqdiff_constant.reset(new simple_cl::cl::Program(kernels::sqdiff_constant_src, kernels::sqdiff_constant_copt, m_cl_context));
				m_program_sqdiff_constant_local.reset(new simple_cl::cl::Program(kernels::sqdiff_constant_local_src, kernels::sqdiff_constant_local_copt, m_cl_context));
				m_program_sqdiff_constant_local_masked.reset(new simple_cl::cl::Program(kernels::sqdiff_constant_local_masked_src, kernels::sqdiff_constant_local_masked_copt, m_cl_context));
				m_program_erode_masked.reset(new simple_cl::cl::Program(kernels::erode_masked_src, kernels::erode_masked_copt, m_cl_context));
				m_program_erode.reset(new simple_cl::cl::Program(kernels::erode_src, kernels::erode_copt, m_cl_context));
				m_program_erode_masked_local.reset(new simple_cl::cl::Program(kernels::erode_masked_local_src, kernels::erode_masked_local_copt, m_cl_context));
				m_program_erode_local.reset(new simple_cl::cl::Program(kernels::erode_local_src, kernels::erode_local_copt, m_cl_context));
				m_program_find_min.reset(new simple_cl::cl::Program(kernels::find_min_src, kernels::find_min_copt, m_cl_context));
				
				// retrieve kernel handles
				m_kernel_naive_sqdiff = m_program_naive_sqdiff->getKernel("sqdiff_naive");
				m_kernel_naive_sqdiff_nth_pass = m_program_naive_sqdiff->getKernel("sqdiff_naive_nth_pass");
				m_kernel_naive_sqdiff_masked = m_program_naive_sqdiff->getKernel("sqdiff_naive_masked");
				m_kernel_naive_sqdiff_masked_nth_pass = m_program_naive_sqdiff->getKernel("sqdiff_naive_masked_nth_pass");

				m_kernel_constant_sqdiff = m_program_sqdiff_constant->getKernel("sqdiff_constant");
				m_kernel_constant_sqdiff_nth_pass = m_program_sqdiff_constant->getKernel("sqdiff_constant_nth_pass");
				m_kernel_constant_sqdiff_masked = m_program_sqdiff_constant->getKernel("sqdiff_constant_masked");
				m_kernel_constant_sqdiff_masked_nth_pass = m_program_sqdiff_constant->getKernel("sqdiff_constant_masked_nth_pass");

				m_kernel_constant_sqdiff_local = m_program_sqdiff_constant_local->getKernel("sqdiff_constant");
				m_kernel_constant_sqdiff_local_nth_pass = m_program_sqdiff_constant_local->getKernel("sqdiff_constant_nth_pass");
				m_kernel_constant_sqdiff_local_masked = m_program_sqdiff_constant_local_masked->getKernel("sqdiff_constant_masked");
				m_kernel_constant_sqdiff_local_masked_nth_pass = m_program_sqdiff_constant_local_masked->getKernel("sqdiff_constant_masked_nth_pass");

				m_kernel_erode_masked = m_program_erode_masked->getKernel("erode");
				m_kernel_erode_constant_masked = m_program_erode_masked->getKernel("erode_constant");
				m_kernel_erode_masked_local = m_program_erode_masked_local->getKernel("erode_constant_local");

				m_kernel_erode = m_program_erode->getKernel("erode");
				m_kernel_erode_local = m_program_erode_local->getKernel("erode_local");

				m_kernel_find_min = m_program_find_min->getKernel("find_min");
				m_kernel_find_min_masked = m_program_find_min->getKernel("find_min_masked");
			}

			inline void ocl_patch_matching::matching_policies::impl::CLMatcherImpl::cleanup_opencl_state()
			{
			}

			inline void ocl_patch_matching::matching_policies::impl::CLMatcherImpl::prepare_input_image(const Texture& input, std::vector<simple_cl::cl::Event>& event_list, bool invalidate, bool blocking)
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
					simple_cl::cl::Image::ImageDimensions{static_cast<std::size_t>(input.response.cols()), static_cast<std::size_t>(input.response.rows()), 1ull}
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
					InputImage& image{m_input_images[tex_index]};
					InputTextureData& texture{image.data};
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
									input.response[static_cast<int>(channel_idx)].convertTo(float_channels[c], CV_32FC1, normalizer[0], normalizer[1]);
								}
								else
								{
									float_channels[c] = cv::Mat(input.response[0].rows, input.response[0].cols, CV_32FC1);
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
					// opencl images
					InputImage input_image;
					input_image.images.reserve(num_images);
					input_image.data.width = static_cast<std::size_t>(input.response.cols());
					input_image.data.height = static_cast<std::size_t>(input.response.rows());
					input_image.data.num_channels = num_feature_maps;
					input_image.data.data.reserve(num_images);

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
								input.response[static_cast<int>(channel_idx)].convertTo(float_channels[c], CV_32FC1, normalizer[0], normalizer[1]);
							}
							else
							{
								float_channels[c] = cv::Mat(input.response[0].rows, input.response[0].cols, CV_32FC1);
								float_channels[c] = cv::Scalar(0.0);
							}
						}
						cv::Mat rgba_img;
						cv::merge(&float_channels[0], 4ull, rgba_img);
						input_image.data.data.push_back(std::move(rgba_img));
					}

					// create images and write image data
					for(std::size_t i{0ull}; i < num_images; ++i)
					{
						// create new image
						input_image.images.push_back(std::unique_ptr<simple_cl::cl::Image>(new simple_cl::cl::Image(m_cl_context, desc)));
						// write new data
						events.push_back(input_image.images.back()->write(img_region, host_fmt,	input_image.data.data[i].data, false));
					}

					// are there free indices?
					if(m_free_indices.empty())
					{
						// add cl image and image data
						m_input_images.push_back(std::move(input_image));
						// get texture index
						std::size_t texture_index{m_input_images.size() - 1};
						// add index to index map
						m_texture_index_map[input.id] = texture_index;
					}
					else // reuse free slot
					{
						// get next free index
						std::size_t texture_index{m_free_indices.top()};
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
						
			inline void ocl_patch_matching::matching_policies::impl::CLMatcherImpl::prepare_texture_mask(const cv::Mat& texture_mask, std::vector<simple_cl::cl::Event>& event_list, bool blocking)
			{
				static cv::Mat mask_data;
				if(mask_data.cols != texture_mask.cols || mask_data.rows != texture_mask.rows)
					mask_data = cv::Mat(texture_mask.rows, texture_mask.cols, CV_32FC1);
				auto normalizer{get_cv_image_normalizer(texture_mask)};
				texture_mask.convertTo(mask_data, CV_32FC1, normalizer[0], normalizer[1]);
				// image region to write
				simple_cl::cl::Image::ImageRegion img_region{
					simple_cl::cl::Image::ImageOffset{0ull, 0ull, 0ull},
					simple_cl::cl::Image::ImageDimensions{static_cast<std::size_t>(texture_mask.cols), static_cast<std::size_t>(texture_mask.rows), 1ull}
				};
				// host format
				simple_cl::cl::Image::HostFormat host_fmt{
					simple_cl::cl::Image::HostChannelOrder{1, {simple_cl::cl::Image::ColorChannel::R, simple_cl::cl::Image::ColorChannel::R, simple_cl::cl::Image::ColorChannel::R, simple_cl::cl::Image::ColorChannel::R}},
					simple_cl::cl::Image::HostDataType::FLOAT,
					simple_cl::cl::Image::HostPitch{static_cast<std::size_t>(mask_data.step[0]), 0ull}
				};
				// if texture mask image nullptr or image is too small, create new one first
				if(!m_texture_mask || !(m_texture_mask->width() >= static_cast<std::size_t>(texture_mask.cols) && m_texture_mask->height() >= static_cast<std::size_t>(texture_mask.rows)))
				{					
					auto desc{make_mask_image_desc(texture_mask)};
					m_texture_mask.reset(new simple_cl::cl::Image(m_cl_context, desc));					
				}
				// upload mask data
				if(blocking)
					m_texture_mask->write(img_region, host_fmt, mask_data.data, true);
				else
					event_list.push_back(m_texture_mask->write(img_region, host_fmt, mask_data.data, false));
			}

			inline void ocl_patch_matching::matching_policies::impl::CLMatcherImpl::prepare_kernel_image(const Texture& kernel_texture, std::vector<simple_cl::cl::Event>& event_list, bool blocking)
			{
				// avoid too many heap allocations
				static std::vector<cv::Mat> kernel_data;
				static cv::Mat float_channels[4];

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
					simple_cl::cl::Image::ImageDimensions{static_cast<std::size_t>(kernel_texture.response.cols()), static_cast<std::size_t>(kernel_texture.response.rows()), 1ull}
				};
				// one input image per 4 feature maps!
				std::size_t num_feature_maps{static_cast<std::size_t>(kernel_texture.response.num_channels())};
				std::size_t num_images{num_feature_maps / 4ull + (num_feature_maps % 4ull != 0ull ? 1ull : 0ull)};
				// normalizer
				auto normalizer{get_cv_image_normalizer(kernel_texture.response[0])};

				// convert new data
				// add cv::Mats for conversion if necessary
				for(std::size_t i{kernel_data.size()}; i < num_images; ++i)
					kernel_data.push_back(cv::Mat());				

				// convert each 4-block of feature maps to float and merge into an rgba image.
				for(std::size_t i{0ull}; i < num_images; ++i)
				{
					for(std::size_t c{0ull}; c < 4ull; ++c)
					{
						std::size_t channel_idx{i * 4ull + c};
						if(channel_idx < num_feature_maps)
						{
							kernel_texture.response[static_cast<int>(channel_idx)].convertTo(float_channels[c], CV_32FC1, normalizer[0], normalizer[1]);
						}
						else
						{
							float_channels[c] = cv::Mat(kernel_texture.response[0].rows, kernel_texture.response[0].cols, CV_32FC1);
							float_channels[c] = cv::Scalar(0.0);
						}
					}
					cv::Mat rgba_img;
					cv::merge(&float_channels[0], 4ull, rgba_img);
					kernel_data[i] = std::move(rgba_img);
				}

				// images not large enough or not yet existent?
				if(!(m_kernel_image.num_channels >= num_feature_maps &&
					m_kernel_image.images[0] &&
					m_kernel_image.images[0]->width() >= static_cast<std::size_t>(kernel_texture.response.cols()) &&
					m_kernel_image.images[0]->height() >= static_cast<std::size_t>(kernel_texture.response.rows())))
				{
					// opencl image desc
					auto desc = make_kernel_image_desc(kernel_texture);
					// create new set of images.
					m_kernel_image.images.clear();
					m_kernel_image.num_channels = num_feature_maps;
					for(std::size_t i{0ull}; i < num_images; ++i)
					{
						m_kernel_image.images.push_back(std::move(std::unique_ptr<simple_cl::cl::Image>(new simple_cl::cl::Image(m_cl_context, desc))));
					}
				}
				// convert and upload new data
				for(std::size_t i{0ull}; i < num_images; ++i)
				{
					events.push_back(std::move(m_kernel_image.images[i]->write(img_region, host_fmt, kernel_data[i].data, false)));
				}
				// wait for upload to finish
				if(blocking)
					simple_cl::cl::wait_for_events(events.begin(), events.end());
				else
					event_list.insert(event_list.end(), events.begin(), events.end());
			}

			inline void ocl_patch_matching::matching_policies::impl::CLMatcherImpl::prepare_kernel_mask(const cv::Mat& kernel_mask, std::vector<simple_cl::cl::Event>& event_list, bool blocking)
			{
				static cv::Mat mask_data;				
				if(mask_data.cols != kernel_mask.cols || mask_data.rows != kernel_mask.rows)
					mask_data = cv::Mat(kernel_mask.rows, kernel_mask.cols, CV_32FC1);
				auto normalizer{get_cv_image_normalizer(kernel_mask)};
				kernel_mask.convertTo(mask_data, CV_32FC1, normalizer[0], normalizer[1]);
				// image region to write
				simple_cl::cl::Image::ImageRegion img_region{
					simple_cl::cl::Image::ImageOffset{0ull, 0ull, 0ull},
					simple_cl::cl::Image::ImageDimensions{static_cast<std::size_t>(kernel_mask.cols), static_cast<std::size_t>(kernel_mask.rows), 1ull}
				};
				// host format
				simple_cl::cl::Image::HostFormat host_fmt{
					simple_cl::cl::Image::HostChannelOrder{1, {simple_cl::cl::Image::ColorChannel::R, simple_cl::cl::Image::ColorChannel::R, simple_cl::cl::Image::ColorChannel::R, simple_cl::cl::Image::ColorChannel::R}},
					simple_cl::cl::Image::HostDataType::FLOAT,
					simple_cl::cl::Image::HostPitch{static_cast<std::size_t>(mask_data.step[0]), 0ull}
				};
				// if kernel mask image nullptr or image is too small, create new one first
				if(!m_kernel_mask || !(m_kernel_mask->width() >= static_cast<std::size_t>(kernel_mask.cols) && m_kernel_mask->height() >= static_cast<std::size_t>(kernel_mask.rows)))
				{
					auto desc{make_kernel_mask_image_desc(kernel_mask)};
					m_kernel_mask.reset(new simple_cl::cl::Image(m_cl_context, desc));
				}
				// upload mask data
				if(blocking)
					m_kernel_mask->write(img_region, host_fmt, mask_data.data, true);
				else
					event_list.push_back(m_kernel_mask->write(img_region, host_fmt, mask_data.data, false));
			}

			inline void ocl_patch_matching::matching_policies::impl::CLMatcherImpl::prepare_kernel_buffer(const Texture& kernel_texture, std::vector<simple_cl::cl::Event>& event_list, bool blocking)
			{
				// avoid too many heap allocations
				static std::vector<cv::Mat> kernel_data;
				static cv::Mat float_channels[4];

				// use async api calls where possible to reduce gpu bubbles. WHY TF DOES OPENCV NOT HAVE MOVE CONTRUCTORS???
				static std::vector<simple_cl::cl::Event> events;
				events.clear();

				// one input image per 4 feature maps!
				std::size_t num_feature_maps{static_cast<std::size_t>(kernel_texture.response.num_channels())};
				std::size_t num_images{num_feature_maps / 4ull + (num_feature_maps % 4ull != 0ull ? 1ull : 0ull)};
				
				// normalizer
				auto normalizer{get_cv_image_normalizer(kernel_texture.response[0])};

				// convert new data
				// add cv::Mats for conversion if necessary
				for(std::size_t i{kernel_data.size()}; i < num_images; ++i)
					kernel_data.push_back(cv::Mat());

				// convert each 4-block of feature maps to float and merge into an rgba image.
				for(std::size_t i{0ull}; i < num_images; ++i)
				{
					for(std::size_t c{0ull}; c < 4ull; ++c)
					{
						std::size_t channel_idx{i * 4ull + c};
						if(channel_idx < num_feature_maps)
						{
							kernel_texture.response[static_cast<int>(channel_idx)].convertTo(float_channels[c], CV_32FC1, normalizer[0], normalizer[1]);
						}
						else
						{
							float_channels[c] = cv::Mat(kernel_texture.response[0].rows, kernel_texture.response[0].cols, CV_32FC1);
							float_channels[c] = cv::Scalar(0.0);
						}
					}
					cv::Mat rgba_img;
					cv::merge(&float_channels[0], 4ull, rgba_img);
					kernel_data[i] = std::move(rgba_img);
				}

				// new buffer size
				std::size_t single_kernel_image_size{static_cast<std::size_t>(kernel_data[0].cols) * static_cast<std::size_t>(kernel_data[0].rows) * sizeof(cl_float4)};
				std::size_t new_buffer_size{kernel_data.size() * single_kernel_image_size};
				// is buffer not yet existing or too small?
				if(!m_kernel_buffer.buffer || m_kernel_buffer.buffer->size() < new_buffer_size)
				{
					simple_cl::cl::MemoryFlags flags{
						simple_cl::cl::DeviceAccess::ReadOnly,
						simple_cl::cl::HostAccess::WriteOnly,
						simple_cl::cl::HostPointerOption::None
					};
					// create new one
					m_kernel_buffer.buffer.reset(new simple_cl::cl::Buffer(new_buffer_size, flags, m_cl_context));
				}
				
				// upload data
				for(std::size_t i{0}; i < kernel_data.size(); ++i)
				{
					events.push_back(std::move(m_kernel_buffer.buffer->write_bytes(kernel_data[i].data, single_kernel_image_size, i * single_kernel_image_size, true)));
				}

				// wait for upload to finish
				if(blocking)
					simple_cl::cl::wait_for_events(events.begin(), events.end());
				else
					event_list.insert(event_list.end(), events.begin(), events.end());
			}

			inline void ocl_patch_matching::matching_policies::impl::CLMatcherImpl::prepare_kernel_mask_buffer(const cv::Mat& kernel_mask, std::vector<simple_cl::cl::Event>& event_list, bool blocking)
			{
				static cv::Mat mask_data;				
				if(mask_data.cols != kernel_mask.cols || mask_data.rows != kernel_mask.rows)
					mask_data = cv::Mat(kernel_mask.rows, kernel_mask.cols, CV_32FC1);
				auto normalizer{get_cv_image_normalizer(kernel_mask)};
				kernel_mask.convertTo(mask_data, CV_32FC1, normalizer[0], normalizer[1]);
				// new buffer size
				std::size_t kernel_mask_size{static_cast<std::size_t>(mask_data.cols) * static_cast<std::size_t>(mask_data.rows) * sizeof(cl_float)};
				// if kernel mask image nullptr or image is too small, create new one first
				if(!m_kernel_mask_buffer.buffer || m_kernel_mask_buffer.buffer->size() < kernel_mask_size)
				{
					simple_cl::cl::MemoryFlags flags{
						simple_cl::cl::DeviceAccess::ReadOnly,
						simple_cl::cl::HostAccess::WriteOnly,
						simple_cl::cl::HostPointerOption::None
					};
					// create new one
					m_kernel_mask_buffer.buffer.reset(new simple_cl::cl::Buffer(kernel_mask_size, flags, m_cl_context));
				}
				// upload mask data
				if(blocking)
					m_kernel_mask_buffer.buffer->write_bytes(mask_data.data, kernel_mask_size, 0ull, true).wait();
				else
					event_list.push_back(std::move(m_kernel_mask_buffer.buffer->write_bytes(mask_data.data, kernel_mask_size, 0ull, true)));
			}

			inline void ocl_patch_matching::matching_policies::impl::CLMatcherImpl::prepare_output_image(const Texture& input, const Texture& kernel, double texture_rotation, const cv::Size& response_dims)
			{
				if(m_output_buffer_a) // if output image already exists
				{
					// recreate output image only if it is too small for the new input - kernel combination
					if(static_cast<std::size_t>(response_dims.width) > m_output_buffer_a->width() || static_cast<std::size_t>(response_dims.height) > m_output_buffer_a->height())
					{
						auto output_desc{make_output_image_desc(input, kernel, texture_rotation, response_dims)};
						m_output_buffer_a.reset(new simple_cl::cl::Image(m_cl_context, output_desc));
					}
				}
				else
				{
					auto output_desc{make_output_image_desc(input, kernel, texture_rotation, response_dims)};
					m_output_buffer_a.reset(new simple_cl::cl::Image(m_cl_context, output_desc));
				}

				// second buffer only needed when we have > 4 feature maps.
				if(input.response.num_channels() > 4)
				{
					if(m_output_buffer_b) // if output image already exists
					{
						// recreate output image only if it is too small for the new input - kernel combination
						if(static_cast<std::size_t>(response_dims.width) > m_output_buffer_b->width() || static_cast<std::size_t>(response_dims.height) > m_output_buffer_b->height())
						{
							auto output_desc{make_output_image_desc(input, kernel, texture_rotation, response_dims)};
							m_output_buffer_b.reset(new simple_cl::cl::Image(m_cl_context, output_desc));
						}
					}
					else
					{
						auto output_desc{make_output_image_desc(input, kernel, texture_rotation, response_dims)};
						m_output_buffer_b.reset(new simple_cl::cl::Image(m_cl_context, output_desc));
					}
				}
			}

			inline void ocl_patch_matching::matching_policies::impl::CLMatcherImpl::prepare_erode_output_image(const cv::Mat& texture_mask)
			{
				if(m_output_texture_mask_eroded) // if output image already exists
				{
					// recreate output image only if it is too small for the new input - kernel combination
					if(static_cast<std::size_t>(texture_mask.cols) > m_output_texture_mask_eroded->width() || static_cast<std::size_t>(texture_mask.rows) > m_output_texture_mask_eroded->height())
					{
						auto output_desc{make_mask_output_image_desc(texture_mask)};
						m_output_texture_mask_eroded.reset(new simple_cl::cl::Image(m_cl_context, output_desc));
					}
				}
				else
				{
					auto output_desc{make_mask_output_image_desc(texture_mask)};
					m_output_texture_mask_eroded.reset(new simple_cl::cl::Image(m_cl_context, output_desc));
				}
			}

			void ocl_patch_matching::matching_policies::impl::CLMatcherImpl::prepare_find_min_output_buffer(const cv::Size& out_size, std::size_t local_work_size_xy, std::size_t& global_work_size_x, std::size_t& global_work_size_y, std::size_t& local_buffer_size)
			{
				// new buffer size
				std::size_t ow{static_cast<size_t>(out_size.width)};
				std::size_t oh{static_cast<size_t>(out_size.height)};
				// multiple of work group width and height
				std::size_t nwg_x{(ow + local_work_size_xy - 1) / local_work_size_xy};
				std::size_t nwg_y{(oh + local_work_size_xy - 1) / local_work_size_xy};
				global_work_size_x = nwg_x * local_work_size_xy;
				global_work_size_y = nwg_y * local_work_size_xy;
				// size of our buffer
				std::size_t new_buffer_size{nwg_x * nwg_y * sizeof(cl_float4)};
				// local buffer size
				local_buffer_size = local_work_size_xy * local_work_size_xy;
				// is buffer not yet existing or too small?
				if(!m_output_buffer_find_min.buffer || m_output_buffer_find_min.buffer->size() < new_buffer_size)
				{
					simple_cl::cl::MemoryFlags flags{
						simple_cl::cl::DeviceAccess::WriteOnly,
						simple_cl::cl::HostAccess::ReadOnly,
						simple_cl::cl::HostPointerOption::None
					};
					// create new one
					m_output_buffer_find_min.buffer.reset(new simple_cl::cl::Buffer(new_buffer_size, flags, m_cl_context));
				}
				m_output_buffer_find_min.num_work_groups[0] = nwg_x;
				m_output_buffer_find_min.num_work_groups[1] = nwg_y;
			}
			
			inline simple_cl::cl::Event CLMatcherImpl::clear_output_image_a(float value)
			{
				simple_cl::cl::Image::ImageRegion region{
					simple_cl::cl::Image::ImageOffset{0ull, 0ull, 0ull},
					simple_cl::cl::Image::ImageDimensions{m_output_buffer_a->width(), m_output_buffer_a->height(), m_output_buffer_a->layers()}
				};
				return m_output_buffer_a->fill(simple_cl::cl::Image::FillColor{value}, region);
			}

			inline simple_cl::cl::Event CLMatcherImpl::clear_output_image_b(float value)
			{
				simple_cl::cl::Image::ImageRegion region{
					simple_cl::cl::Image::ImageOffset{0ull, 0ull, 0ull},
					simple_cl::cl::Image::ImageDimensions{m_output_buffer_b->width(), m_output_buffer_b->height(), m_output_buffer_b->layers()}
				};
				return m_output_buffer_b->fill(simple_cl::cl::Image::FillColor{value}, region);
			}

			inline bool ocl_patch_matching::matching_policies::impl::CLMatcherImpl::use_constant_kernel(const Texture& kernel, const cv::Mat& kernel_mask) const
			{
				std::size_t num_feature_maps{static_cast<std::size_t>(kernel.response.num_channels())};
				std::size_t num_batches{num_feature_maps / 4ull + (num_feature_maps % 4ull != 0ull ? 1ull : 0ull)};
				std::size_t kernel_pixels{static_cast<std::size_t>(kernel.response.cols()) * static_cast<std::size_t>(kernel.response.rows())};
				std::size_t total_size{(sizeof(cl_float4) + sizeof(cl_float)) * kernel_pixels * num_batches};
				return (kernel_pixels <= m_constant_kernel_max_pixels && total_size <= m_cl_context->get_selected_device().max_constant_buffer_size);
			}

			inline bool ocl_patch_matching::matching_policies::impl::CLMatcherImpl::use_constant_kernel(const Texture& kernel) const
			{
				std::size_t num_feature_maps{static_cast<std::size_t>(kernel.response.num_channels())};
				std::size_t num_batches{num_feature_maps / 4ull + (num_feature_maps % 4ull != 0ull ? 1ull : 0ull)};
				std::size_t kernel_pixels{static_cast<std::size_t>(kernel.response.cols()) * static_cast<std::size_t>(kernel.response.rows())};
				std::size_t total_size{(sizeof(cl_float4)) * kernel_pixels * num_batches};
				return (kernel_pixels <= m_constant_kernel_max_pixels && total_size <= m_cl_context->get_selected_device().max_constant_buffer_size);
			}

			inline bool ocl_patch_matching::matching_policies::impl::CLMatcherImpl::use_constant_kernel(const cv::Mat& kernel_mask) const
			{
				std::size_t kernel_pixels{static_cast<std::size_t>(kernel_mask.cols) * static_cast<std::size_t>(kernel_mask.cols)};
				std::size_t total_size{sizeof(cl_float) * kernel_pixels};
				return (kernel_pixels <= m_constant_kernel_max_pixels && total_size <= m_cl_context->get_selected_device().max_constant_buffer_size);
			}
			
			inline bool ocl_patch_matching::matching_policies::impl::CLMatcherImpl::use_local_mem(const cv::Vec4i& kernel_overlaps, std::size_t used_local_mem, std::size_t local_work_size, std::size_t max_pixels, std::size_t size_per_pixel)
			{
				std::size_t max_overlap(static_cast<std::size_t>(std::max({kernel_overlaps[0], kernel_overlaps[1], kernel_overlaps[2], kernel_overlaps[3]})));
				std::size_t num_pixels{static_cast<std::size_t>(kernel_overlaps[0] + local_work_size + kernel_overlaps[1]) * static_cast<std::size_t>(kernel_overlaps[2] + local_work_size + kernel_overlaps[3])};
				std::size_t total_size{num_pixels * size_per_pixel};
				return (
					num_pixels <= max_pixels && // less than num pixels threshold? (for control from outside)
					total_size <= (m_cl_context->get_selected_device().local_mem_size - used_local_mem) && // less than or equal to free local memory?
					max_overlap <= local_work_size // max overlap less than or equal to the local work size (work group diameter!)? If not we cannot load all pixels into local memory.
					);
			}

			inline std::size_t ocl_patch_matching::matching_policies::impl::CLMatcherImpl::get_local_work_size(const simple_cl::cl::Program::CLKernelHandle& kernel) const
			{
				std::size_t wgsize{m_local_block_size};
				while(!(wgsize * wgsize <= kernel.getKernelInfo().max_work_group_size))
					wgsize /= 2ull;
				return wgsize;
			}
			
			inline void ocl_patch_matching::matching_policies::impl::CLMatcherImpl::calculate_rotated_kernel_dims(cv::Size& rotated_kernel_size, cv::Vec4i& rotated_kernel_overlaps, const Texture& kernel, double texture_rotation, const cv::Point& anchor)
			{
				float pivot_x{0.0f};
				float pivot_y{0.0f};
				if(anchor.x == -1 && anchor.y == -1)
				{
					pivot_x = static_cast<float>((kernel.response.cols() - 1) / 2) + 0.5f;
					pivot_y = static_cast<float>((kernel.response.rows() - 1) / 2) + 0.5f;
				}
				else
				{
					pivot_x = static_cast<float>(anchor.x) + 0.5f;
					pivot_y = static_cast<float>(anchor.y) + 0.5f;
				}
				// two corners (sampling coordinates!) of the unrotated kernel
				float top_left_x = 0.5f - pivot_x;
				float top_left_y = 0.5f - pivot_y;
				float bottom_right_x = static_cast<float>(kernel.response.cols() - 1) + 0.5f - pivot_x;
				float bottom_right_y = static_cast<float>(kernel.response.rows() - 1) + 0.5f - pivot_y;

				// cosine and sine of the rotation angle (texture rotates negatively => we must positively rotate the samples)
				float c = cosf(static_cast<float>(texture_rotation));
				float s = sinf(static_cast<float>(texture_rotation));

				// four rotated corners
				float rotated_top_left_x = c * top_left_x - s * top_left_y;
				float rotated_top_left_y = s * top_left_x + c * top_left_y;

				float rotated_top_right_x = c * bottom_right_x - s * top_left_y;
				float rotated_top_right_y = s * bottom_right_x + c * top_left_y;

				float rotated_bottom_left_x = c * top_left_x - s * bottom_right_y;
				float rotated_bottom_left_y = s * top_left_x + c * bottom_right_y;

				float rotated_bottom_right_x = c * bottom_right_x - s * bottom_right_y;
				float rotated_bottom_right_y = s * bottom_right_x + c * bottom_right_y;

				// min and max sample coords for kernel bounding box
				float min_x = std::min({rotated_top_left_x, rotated_top_right_x, rotated_bottom_left_x, rotated_bottom_right_x});
				float min_y = std::min({rotated_top_left_y, rotated_top_right_y, rotated_bottom_left_y, rotated_bottom_right_y});
				float max_x = std::max({rotated_top_left_x, rotated_top_right_x, rotated_bottom_left_x, rotated_bottom_right_x});
				float max_y = std::max({rotated_top_left_y, rotated_top_right_y, rotated_bottom_left_y, rotated_bottom_right_y});

				// rotated bounding box width / height in pixels
				int rbb_width = static_cast<int>(floorf(max_x)) - static_cast<int>(floorf(min_x)) + 1;
				int rbb_height = static_cast<int>(floorf(max_y)) - static_cast<int>(floorf(min_y)) + 1;

				// new pivot of the rotated thingy
				int new_pivot_x = static_cast<int>(floorf(-min_x + 0.5f));
				int new_pivot_y = static_cast<int>(floorf(-min_y + 0.5f));

				// overlaps of the new kernel thingy
				// left
				rotated_kernel_overlaps[0] = new_pivot_x;
				// right
				rotated_kernel_overlaps[1] = rbb_width - 1 - new_pivot_x;
				// top
				rotated_kernel_overlaps[2] = new_pivot_y;
				// bottom
				rotated_kernel_overlaps[3] = rbb_height - 1 - new_pivot_y;
				// rotated kernel size
				rotated_kernel_size.width = rbb_width;
				rotated_kernel_size.height = rbb_height;
			}
			
			inline simple_cl::cl::Event ocl_patch_matching::matching_policies::impl::CLMatcherImpl::read_output_image(cv::Mat& out_mat, const cv::Size& output_size, const std::vector<simple_cl::cl::Event>& wait_for, bool out_a)
			{
				// resize output if necessary
				if((output_size.width != out_mat.cols) ||
					(output_size.height != out_mat.rows))
				{
					out_mat = cv::Mat(output_size.height, output_size.width, CV_32FC1);
				}

				// image region
				simple_cl::cl::Image::ImageRegion region{
					simple_cl::cl::Image::ImageOffset{0ull, 0ull, 0ull},
					simple_cl::cl::Image::ImageDimensions{static_cast<std::size_t>(output_size.width), static_cast<std::size_t>(output_size.height), 1ull}
				};

				// host format
				simple_cl::cl::Image::HostFormat hostfmt{
					simple_cl::cl::Image::HostChannelOrder{1ull, {simple_cl::cl::Image::ColorChannel::R, simple_cl::cl::Image::ColorChannel::R, simple_cl::cl::Image::ColorChannel::R, simple_cl::cl::Image::ColorChannel::R}},
					simple_cl::cl::Image::HostDataType::FLOAT,
					simple_cl::cl::Image::HostPitch{static_cast<std::size_t>(out_mat.step[0]), 0ull}
				};

				// read output and return event
				if(out_a)
					return m_output_buffer_a->read(region, hostfmt, out_mat.data, wait_for.begin(), wait_for.end());
				else
					return m_output_buffer_b->read(region, hostfmt, out_mat.data, wait_for.begin(), wait_for.end());
			}

			inline simple_cl::cl::Event ocl_patch_matching::matching_policies::impl::CLMatcherImpl::read_eroded_texture_mask_image(cv::Mat& out_mat, const cv::Size& output_size, const std::vector<simple_cl::cl::Event>& wait_for)
			{
				// resize output if necessary
				if((output_size.width != out_mat.cols) || (output_size.height != out_mat.rows))
				{
					out_mat = cv::Mat(output_size.height, output_size.width, CV_32FC1);
				}

				// image region
				simple_cl::cl::Image::ImageRegion region{
					simple_cl::cl::Image::ImageOffset{0ull, 0ull, 0ull},
					simple_cl::cl::Image::ImageDimensions{static_cast<std::size_t>(output_size.width), static_cast<std::size_t>(output_size.height), 1ull}
				};

				// host format
				simple_cl::cl::Image::HostFormat hostfmt{
					simple_cl::cl::Image::HostChannelOrder{1ull, {simple_cl::cl::Image::ColorChannel::R, simple_cl::cl::Image::ColorChannel::R, simple_cl::cl::Image::ColorChannel::R, simple_cl::cl::Image::ColorChannel::R}},
					simple_cl::cl::Image::HostDataType::FLOAT,
					simple_cl::cl::Image::HostPitch{static_cast<std::size_t>(out_mat.step[0]), 0ull}
				};

				// read output and return event				
				return m_output_texture_mask_eroded->read(region, hostfmt, out_mat.data, wait_for.begin(), wait_for.end());
			}
			
			void ocl_patch_matching::matching_policies::impl::CLMatcherImpl::read_min_pos_and_cost(MatchingResult& res, const std::vector<simple_cl::cl::Event>& wait_for, const cv::Point& res_coord_offset)
			{
				static std::vector<cl_float4> work_group_results;
				if(work_group_results.size() != (m_output_buffer_find_min.num_work_groups[0] * m_output_buffer_find_min.num_work_groups[1]))
				{
					work_group_results.resize(m_output_buffer_find_min.num_work_groups[0] * m_output_buffer_find_min.num_work_groups[1], cl_float4{std::numeric_limits<float>::max(), 0.0f, 0.0f, 0.0f});
				}				

				// read results from buffer
				m_output_buffer_find_min.buffer->read(work_group_results.begin(), m_output_buffer_find_min.num_work_groups[0] * m_output_buffer_find_min.num_work_groups[1], wait_for.begin(), wait_for.end()).wait();
				// find minimum
				cl_float4 minimum{*std::min_element(work_group_results.begin(), work_group_results.end(), [](const cl_float4& lhs, const cl_float4& rhs) { return lhs.x < rhs.x; })};
				// write min position and cost
				res.matches.clear();
				res.matches.push_back(Match{cv::Point(static_cast<int>(std::floorf(minimum.z)) + res_coord_offset.x, static_cast<int>(std::floorf(minimum.w)) + res_coord_offset.y), minimum.x});				
			}			
			
			inline void ocl_patch_matching::matching_policies::impl::CLMatcherImpl::compute_matches(
				const Texture& texture,
				const Texture& kernel,
				const cv::Mat& kernel_mask,
				double texture_rotation,
				MatchingResult& match_res_out)
			{
				static std::vector<simple_cl::cl::Event> pre_compute_events;
				pre_compute_events.clear();
				// compute rotated kernel size
				// kernel anchor
				cv::Point kernel_anchor{(m_result_origin == CLMatcher::ResultOrigin::Center ? cv::Point((kernel.response.cols() - 1) / 2, (kernel.response.rows() - 1) / 2) : cv::Point(0, 0))};
				cv::Size rotated_kernel_size;
				cv::Vec4i rotated_kernel_overlaps;
				calculate_rotated_kernel_dims(rotated_kernel_size, rotated_kernel_overlaps, kernel, texture_rotation, kernel_anchor);
				cv::Size response_dims = get_response_dimensions(texture, kernel, texture_rotation, kernel_anchor);
				// prepare all input data
				prepare_input_image(texture, pre_compute_events, false, false);
				bool use_constant{use_constant_kernel(kernel, kernel_mask)};
				if(use_constant)
				{
					prepare_kernel_buffer(kernel, pre_compute_events, false);
					prepare_kernel_mask_buffer(kernel_mask, pre_compute_events, false);
				}
				else
				{
					prepare_kernel_image(kernel, pre_compute_events, false);
					prepare_kernel_mask(kernel_mask, pre_compute_events, false);
				}				
				prepare_output_image(texture, kernel, texture_rotation, response_dims);
				// get input image from map
				InputImage& input_image{m_input_images[m_texture_index_map[texture.id]]};
				// pingpong between the two output buffers until done
				std::size_t num_feature_maps{static_cast<std::size_t>(texture.response.num_channels())};
				std::size_t num_batches{num_feature_maps / 4ull + (num_feature_maps % 4ull != 0ull ? 1ull : 0ull)};
				// exec params
				simple_cl::cl::Program::ExecParams exec_params{
					2ull,
					{0ull, 0ull, 0ull},
					{static_cast<std::size_t>(response_dims.width), static_cast<std::size_t>(response_dims.height), 1ull},
					{m_local_block_size, m_local_block_size, 1ull}
				};
				if(!use_constant)
				{
					// calc local work group size
					std::size_t wg_size{std::min(get_local_work_size(m_kernel_naive_sqdiff_masked), get_local_work_size(m_kernel_naive_sqdiff_masked_nth_pass))};
					exec_params.local_work_size[0] = wg_size;
					exec_params.local_work_size[1] = wg_size;
					// other arguments
					cl_int2 input_size{texture.response.cols(), texture.response.rows()};
					cl_int2 kernel_size{kernel.response.cols(), kernel.response.rows()};
					cl_int2 input_piv{rotated_kernel_overlaps[0], rotated_kernel_overlaps[2]};
					cl_float2 rotation_sincos{std::sinf(static_cast<float>(texture_rotation)), std::cosf(static_cast<float>(texture_rotation))};
					// first pass
					simple_cl::cl::Event first_event{(*m_program_naive_sqdiff)(
						m_kernel_naive_sqdiff_masked,
						pre_compute_events.begin(),
						pre_compute_events.end(),
						exec_params,
						*(input_image.images[0]),
						*(m_kernel_image.images[0]),
						*(m_kernel_mask),
						*(m_output_buffer_a),
						input_size,
						kernel_size,
						cl_int2{kernel_anchor.x, kernel_anchor.y},
						input_piv,
						rotation_sincos)
					};
					pre_compute_events.clear();
					pre_compute_events.push_back(std::move(first_event));
					// if necessary, more passes
					for(std::size_t batch{1}; batch < num_batches; ++batch) // ping pong between two output buffers
					{
						if(batch % 2ull == 0)
						{
							simple_cl::cl::Event event{(*m_program_naive_sqdiff)(
								m_kernel_naive_sqdiff_masked_nth_pass,
								pre_compute_events.begin(),
								pre_compute_events.end(),
								exec_params,
								*(input_image.images[batch]),
								*(m_kernel_image.images[batch]),
								*(m_kernel_mask),
								*(m_output_buffer_b),
								*(m_output_buffer_a),
								input_size,
								kernel_size,
								cl_int2{kernel_anchor.x, kernel_anchor.y},
								input_piv,
								rotation_sincos)
							};
							pre_compute_events.clear();
							pre_compute_events.push_back(std::move(event));
						}
						else
						{
							simple_cl::cl::Event event{(*m_program_naive_sqdiff)(
								m_kernel_naive_sqdiff_masked_nth_pass,
								pre_compute_events.begin(),
								pre_compute_events.end(),
								exec_params,
								*(input_image.images[batch]),
								*(m_kernel_image.images[batch]),
								*(m_kernel_mask),
								*(m_output_buffer_a),
								*(m_output_buffer_b),
								input_size,
								kernel_size,
								cl_int2{kernel_anchor.x, kernel_anchor.y},
								input_piv,
								rotation_sincos)
							};
							pre_compute_events.clear();
							pre_compute_events.push_back(std::move(event));
						}
					}
				}
				else
				{
					// get safe local wg size
					std::size_t wg_size{std::min(get_local_work_size(m_kernel_constant_sqdiff_masked), get_local_work_size(m_kernel_constant_sqdiff_masked_nth_pass))};
					std::size_t wg_size_local{std::min(get_local_work_size(m_kernel_constant_sqdiff_local_masked), get_local_work_size(m_kernel_constant_sqdiff_local_masked_nth_pass))};

					// calculate total buffer size in pixels
					std::size_t local_buffer_total_size{static_cast<std::size_t>(rotated_kernel_overlaps[0] + wg_size_local + rotated_kernel_overlaps[1]) * static_cast<std::size_t>(rotated_kernel_overlaps[2] + wg_size_local + rotated_kernel_overlaps[3])};

					// decide if we should use the version with local memory optimization
					std::size_t wg_used_local_mem{std::max(m_kernel_constant_sqdiff_local_masked.getKernelInfo().local_memory_usage, m_kernel_constant_sqdiff_local_masked_nth_pass.getKernelInfo().local_memory_usage)};					
					bool use_local{use_local_mem(rotated_kernel_overlaps, wg_used_local_mem, wg_size_local, m_local_buffer_max_pixels, sizeof(cl_float4)) && m_use_local_buffer_for_matching};
					if(!use_local)
					{
						exec_params.local_work_size[0] = wg_size;
						exec_params.local_work_size[1] = wg_size;
						// other arguments
						cl_int2 input_size{texture.response.cols(), texture.response.rows()};
						cl_int2 kernel_size{kernel.response.cols(), kernel.response.rows()};
						cl_int2 input_piv{rotated_kernel_overlaps[0], rotated_kernel_overlaps[2]};
						cl_float2 rotation_sincos{std::sinf(static_cast<float>(texture_rotation)), std::cosf(static_cast<float>(texture_rotation))};
						cl_int kernel_offset{0};
						cl_int num_kernel_pixels{kernel.response.cols() * kernel.response.rows()};
						// first pass
						simple_cl::cl::Event first_event{(*m_program_sqdiff_constant)(
							m_kernel_constant_sqdiff_masked,
							pre_compute_events.begin(),
							pre_compute_events.end(),
							exec_params,
							*(input_image.images[0]),
							*(m_kernel_buffer.buffer),
							*(m_kernel_mask_buffer.buffer),
							*(m_output_buffer_a),
							input_size,
							kernel_size,
							cl_int2{kernel_anchor.x, kernel_anchor.y},
							input_piv,
							rotation_sincos)
						};
						pre_compute_events.clear();
						pre_compute_events.push_back(std::move(first_event));
						// if necessary, more passes
						for(std::size_t batch{1}; batch < num_batches; ++batch) // ping pong between two output buffers
						{
							kernel_offset += num_kernel_pixels;
							if(batch % 2ull == 0)
							{
								simple_cl::cl::Event event{(*m_program_sqdiff_constant)(
									m_kernel_constant_sqdiff_masked_nth_pass,
									pre_compute_events.begin(),
									pre_compute_events.end(),
									exec_params,
									*(input_image.images[batch]),
									*(m_kernel_buffer.buffer),
									*(m_kernel_mask_buffer.buffer),
									*(m_output_buffer_b),
									*(m_output_buffer_a),
									input_size,
									kernel_size,
									cl_int2{kernel_anchor.x, kernel_anchor.y},
									input_piv,
									rotation_sincos,
									kernel_offset)
								};
								pre_compute_events.clear();
								pre_compute_events.push_back(std::move(event));
							}
							else
							{
								simple_cl::cl::Event event{(*m_program_sqdiff_constant)(
									m_kernel_constant_sqdiff_masked_nth_pass,
									pre_compute_events.begin(),
									pre_compute_events.end(),
									exec_params,
									*(input_image.images[batch]),
									*(m_kernel_buffer.buffer),
									*(m_kernel_mask_buffer.buffer),
									*(m_output_buffer_a),
									*(m_output_buffer_b),
									input_size,
									kernel_size,
									cl_int2{kernel_anchor.x, kernel_anchor.y},
									input_piv,
									rotation_sincos,
									kernel_offset)
								};
								pre_compute_events.clear();
								pre_compute_events.push_back(std::move(event));
							}
						}
					}
					else
					{		
						exec_params.local_work_size[0] = wg_size_local;
						exec_params.local_work_size[1] = wg_size_local;
						// pad global work size to a multiple of the work group size.
						// this is necessary to make the local memory scheme work.
						exec_params.global_work_size[0] = ((exec_params.global_work_size[0] + wg_size_local - 1) / wg_size_local) * wg_size_local;
						exec_params.global_work_size[1] = ((exec_params.global_work_size[1] + wg_size_local - 1) / wg_size_local) * wg_size_local;
						// other arguments
						cl_int2 input_size{texture.response.cols(), texture.response.rows()};
						cl_int2 output_size{response_dims.width, response_dims.height};
						cl_int2 kernel_size{kernel.response.cols(), kernel.response.rows()};
						cl_int2 input_piv{rotated_kernel_overlaps[0], rotated_kernel_overlaps[2]};
						cl_float2 rotation_sincos{std::sinf(static_cast<float>(texture_rotation)), std::cosf(static_cast<float>(texture_rotation))};
						cl_int kernel_offset{0};
						cl_int num_kernel_pixels{kernel.response.cols() * kernel.response.rows()};
						// first pass
						simple_cl::cl::Event first_event{(*m_program_sqdiff_constant_local_masked)(
							m_kernel_constant_sqdiff_local_masked,
							pre_compute_events.begin(),
							pre_compute_events.end(),
							exec_params,
							*(input_image.images[0]),
							simple_cl::cl::LocalMemory<cl_float4>(local_buffer_total_size),
							*(m_kernel_buffer.buffer),
							*(m_kernel_mask_buffer.buffer),
							*(m_output_buffer_a),
							input_size,
							output_size,
							kernel_size,
							cl_int2{kernel_anchor.x, kernel_anchor.y},
							input_piv,
							cl_int4{rotated_kernel_overlaps[0], rotated_kernel_overlaps[1], rotated_kernel_overlaps[2], rotated_kernel_overlaps[3]},
							rotation_sincos)
						};
						pre_compute_events.clear();
						pre_compute_events.push_back(std::move(first_event));
						// if necessary, more passes
						for(std::size_t batch{1}; batch < num_batches; ++batch) // ping pong between two output buffers
						{
							kernel_offset += num_kernel_pixels;
							if(batch % 2ull == 0)
							{
								simple_cl::cl::Event event{(*m_program_sqdiff_constant_local_masked)(
									m_kernel_constant_sqdiff_local_masked_nth_pass,
									pre_compute_events.begin(),
									pre_compute_events.end(),
									exec_params,
									*(input_image.images[batch]),
									simple_cl::cl::LocalMemory<cl_float4>(local_buffer_total_size),
									*(m_kernel_buffer.buffer),
									*(m_kernel_mask_buffer.buffer),
									*(m_output_buffer_b),
									*(m_output_buffer_a),
									input_size,
									output_size,
									kernel_size,
									cl_int2{kernel_anchor.x, kernel_anchor.y},
									input_piv,
									cl_int4{rotated_kernel_overlaps[0], rotated_kernel_overlaps[1], rotated_kernel_overlaps[2], rotated_kernel_overlaps[3]},
									rotation_sincos,
									kernel_offset)
								};
								pre_compute_events.clear();
								pre_compute_events.push_back(std::move(event));
							}
							else
							{
								simple_cl::cl::Event event{(*m_program_sqdiff_constant_local_masked)(
									m_kernel_constant_sqdiff_local_masked_nth_pass,
									pre_compute_events.begin(),
									pre_compute_events.end(),
									exec_params,
									*(input_image.images[batch]),
									simple_cl::cl::LocalMemory<cl_float4>(local_buffer_total_size),
									*(m_kernel_buffer.buffer),
									*(m_kernel_mask_buffer.buffer),
									*(m_output_buffer_a),
									*(m_output_buffer_b),
									input_size,
									output_size,
									kernel_size,
									cl_int2{kernel_anchor.x, kernel_anchor.y},
									input_piv,
									cl_int4{rotated_kernel_overlaps[0], rotated_kernel_overlaps[1], rotated_kernel_overlaps[2], rotated_kernel_overlaps[3]},
									rotation_sincos,
									kernel_offset)
								};
								pre_compute_events.clear();
								pre_compute_events.push_back(std::move(event));
							}
						}
					}
				}
				// prepare stuff for minimum extraction while kernel is running
				std::size_t find_min_local_work_size{get_local_work_size(m_kernel_find_min)};
				simple_cl::cl::Program::ExecParams find_min_exec_params{
					2ull,
					{0ull, 0ull, 0ull},
					{0ull, 0ull, 1ull},
					{find_min_local_work_size, find_min_local_work_size, 1ull}
				};
				std::size_t find_min_local_buffer_size;
				prepare_find_min_output_buffer(cv::Size{response_dims.width, response_dims.height}, find_min_local_work_size, find_min_exec_params.global_work_size[0], find_min_exec_params.global_work_size[1], find_min_local_buffer_size);

				// read result and wait for command chain to finish execution. If num_batches is odd, read output a, else b.
				simple_cl::cl::Event response_finished_event{read_output_image(match_res_out.total_cost_matrix, response_dims, pre_compute_events, num_batches % 2ull)};
				pre_compute_events.clear();
				pre_compute_events.push_back(std::move(response_finished_event));

				// run kernel for minimum extraction
				simple_cl::cl::Event find_min_kernel_event{(*m_program_find_min)(
					m_kernel_find_min,
					pre_compute_events.begin(),
					pre_compute_events.end(),
					find_min_exec_params,
					(num_batches % 2ull ? *m_output_buffer_a : *m_output_buffer_b),
					*(m_output_buffer_find_min.buffer),
					simple_cl::cl::LocalMemory<cl_float4>(find_min_local_buffer_size),
					cl_int2{response_dims.width, response_dims.height})
				};
				pre_compute_events.clear();
				pre_compute_events.push_back(std::move(find_min_kernel_event));
				// output result
				cv::Point result_offset = cv::Point(rotated_kernel_overlaps[0], rotated_kernel_overlaps[2]);
				read_min_pos_and_cost(match_res_out, pre_compute_events, result_offset);
			}

			inline void ocl_patch_matching::matching_policies::impl::CLMatcherImpl::compute_matches(
				const Texture& texture,
				const Texture& kernel,
				double texture_rotation,
				MatchingResult& match_res_out)
			{
				static std::vector<simple_cl::cl::Event> pre_compute_events;
				pre_compute_events.clear();
				// compute rotated kernel size
				// kernel anchor
				cv::Point kernel_anchor{(m_result_origin == CLMatcher::ResultOrigin::Center ? cv::Point((kernel.response.cols() - 1) / 2, (kernel.response.rows() - 1) / 2) : cv::Point(0, 0))};
				cv::Size rotated_kernel_size;
				cv::Vec4i rotated_kernel_overlaps;
				calculate_rotated_kernel_dims(rotated_kernel_size, rotated_kernel_overlaps, kernel, texture_rotation, kernel_anchor);
				cv::Size response_dims = get_response_dimensions(texture, kernel, texture_rotation, kernel_anchor);
				// prepare all input data
				// upload input image
				prepare_input_image(texture, pre_compute_events, false, false);
				// upload kernel data
				bool use_constant{use_constant_kernel(kernel)};
				if(use_constant)
					prepare_kernel_buffer(kernel, pre_compute_events, false);
				else
					prepare_kernel_image(kernel, pre_compute_events, false);
				// output images
				prepare_output_image(texture, kernel, texture_rotation, response_dims);
				// get input image from map
				InputImage& input_image{m_input_images[m_texture_index_map[texture.id]]};
				// pingpong between the two output buffers until done
				std::size_t num_feature_maps{static_cast<std::size_t>(texture.response.num_channels())};
				std::size_t num_batches{num_feature_maps / 4ull + (num_feature_maps % 4ull != 0ull ? 1ull : 0ull)};
				// exec params
				simple_cl::cl::Program::ExecParams exec_params{
					2ull,
					{0ull, 0ull, 0ull},
					{static_cast<std::size_t>(response_dims.width), static_cast<std::size_t>(response_dims.height), 1ull},
					{m_local_block_size, m_local_block_size, 1ull}
				};
				if(!use_constant)
				{
					std::size_t wg_size{std::min(get_local_work_size(m_kernel_naive_sqdiff), get_local_work_size(m_kernel_naive_sqdiff_nth_pass))};
					exec_params.local_work_size[0] = wg_size;
					exec_params.local_work_size[1] = wg_size;
					// other arguments
					cl_int2 input_size{texture.response.cols(), texture.response.rows()};
					cl_int2 kernel_size{kernel.response.cols(), kernel.response.rows()};
					cl_int2 input_piv{rotated_kernel_overlaps[0], rotated_kernel_overlaps[2]};
					cl_float2 rotation_sincos{std::sinf(static_cast<float>(texture_rotation)), std::cosf(static_cast<float>(texture_rotation))};
					// first pass
					simple_cl::cl::Event first_event{(*m_program_naive_sqdiff)(
						m_kernel_naive_sqdiff,
						pre_compute_events.begin(),
						pre_compute_events.end(),
						exec_params,
						*(input_image.images[0]),
						*(m_kernel_image.images[0]),
						*(m_output_buffer_a),
						input_size,
						kernel_size,
						cl_int2{kernel_anchor.x, kernel_anchor.y},
						input_piv,
						rotation_sincos)
					};
					pre_compute_events.clear();
					pre_compute_events.push_back(std::move(first_event));
					// if necessary, more passes
					for(std::size_t batch{1}; batch < num_batches; ++batch) // ping pong between two output buffers
					{
						if(batch % 2ull == 0)
						{
							simple_cl::cl::Event event{(*m_program_naive_sqdiff)(
								m_kernel_naive_sqdiff_nth_pass,
								pre_compute_events.begin(),
								pre_compute_events.end(),
								exec_params,
								*(input_image.images[batch]),
								*(m_kernel_image.images[batch]),
								*(m_output_buffer_b),
								*(m_output_buffer_a),
								input_size,
								kernel_size,
								cl_int2{kernel_anchor.x, kernel_anchor.y},
								input_piv,
								rotation_sincos)
							};
							pre_compute_events.clear();
							pre_compute_events.push_back(std::move(event));
						}
						else
						{
							simple_cl::cl::Event event{(*m_program_naive_sqdiff)(
								m_kernel_naive_sqdiff_nth_pass,
								pre_compute_events.begin(),
								pre_compute_events.end(),
								exec_params,
								*(input_image.images[batch]),
								*(m_kernel_image.images[batch]),
								*(m_output_buffer_a),
								*(m_output_buffer_b),
								input_size,
								kernel_size,
								cl_int2{kernel_anchor.x, kernel_anchor.y},
								input_piv,
								rotation_sincos)
							};
							pre_compute_events.clear();
							pre_compute_events.push_back(std::move(event));
						}
					}
				}
				else
				{					
					// get safe local wg size
					std::size_t wg_size{std::min(get_local_work_size(m_kernel_constant_sqdiff), get_local_work_size(m_kernel_constant_sqdiff_nth_pass))};
					std::size_t wg_size_local{std::min(get_local_work_size(m_kernel_constant_sqdiff_local), get_local_work_size(m_kernel_constant_sqdiff_local_nth_pass))};
					std::size_t wg_used_local_mem{std::max(m_kernel_constant_sqdiff_local.getKernelInfo().local_memory_usage, m_kernel_constant_sqdiff_local_nth_pass.getKernelInfo().local_memory_usage)};

					// calculate total buffer size in pixels
					std::size_t local_buffer_total_size{static_cast<std::size_t>(rotated_kernel_overlaps[0] + wg_size_local + rotated_kernel_overlaps[1]) * static_cast<std::size_t>(rotated_kernel_overlaps[2] + wg_size_local + rotated_kernel_overlaps[3])};

					// decide if we should use local memory optimizatio
					bool use_local{use_local_mem(rotated_kernel_overlaps, wg_used_local_mem, wg_size_local, m_local_buffer_max_pixels, sizeof(cl_float4)) && m_use_local_buffer_for_matching};

					if(!use_local)
					{
						exec_params.local_work_size[0] = wg_size;
						exec_params.local_work_size[1] = wg_size;
						// other arguments
						cl_int2 input_size{texture.response.cols(), texture.response.rows()};
						cl_int2 kernel_size{kernel.response.cols(), kernel.response.rows()};
						cl_int2 input_piv{rotated_kernel_overlaps[0], rotated_kernel_overlaps[2]};
						cl_float2 rotation_sincos{std::sinf(static_cast<float>(texture_rotation)), std::cosf(static_cast<float>(texture_rotation))};
						cl_int kernel_offset{0};
						cl_int num_kernel_pixels{kernel.response.cols() * kernel.response.rows()};
						// first pass
						simple_cl::cl::Event first_event{(*m_program_sqdiff_constant)(
							m_kernel_constant_sqdiff,
							pre_compute_events.begin(),
							pre_compute_events.end(),
							exec_params,
							*(input_image.images[0]),
							*(m_kernel_buffer.buffer),
							*(m_output_buffer_a),
							input_size,
							kernel_size,
							cl_int2{kernel_anchor.x, kernel_anchor.y},
							input_piv,
							rotation_sincos)
						};
						pre_compute_events.clear();
						pre_compute_events.push_back(std::move(first_event));
						// if necessary, more passes
						for(std::size_t batch{1}; batch < num_batches; ++batch) // ping pong between two output buffers
						{
							kernel_offset += num_kernel_pixels;
							if(batch % 2ull == 0)
							{
								simple_cl::cl::Event event{(*m_program_sqdiff_constant)(
									m_kernel_constant_sqdiff_nth_pass,
									pre_compute_events.begin(),
									pre_compute_events.end(),
									exec_params,
									*(input_image.images[batch]),
									*(m_kernel_buffer.buffer),
									*(m_output_buffer_b),
									*(m_output_buffer_a),
									input_size,
									kernel_size,
									cl_int2{kernel_anchor.x, kernel_anchor.y},
									input_piv,
									rotation_sincos,
									kernel_offset)
								};
								pre_compute_events.clear();
								pre_compute_events.push_back(std::move(event));
							}
							else
							{
								simple_cl::cl::Event event{(*m_program_sqdiff_constant)(
									m_kernel_constant_sqdiff_nth_pass,
									pre_compute_events.begin(),
									pre_compute_events.end(),
									exec_params,
									*(input_image.images[batch]),
									*(m_kernel_buffer.buffer),
									*(m_output_buffer_a),
									*(m_output_buffer_b),
									input_size,
									kernel_size,
									cl_int2{kernel_anchor.x, kernel_anchor.y},
									input_piv,
									rotation_sincos,
									kernel_offset)
								};
								pre_compute_events.clear();
								pre_compute_events.push_back(std::move(event));
							}
						}
					}
					else
					{
						exec_params.local_work_size[0] = wg_size_local;
						exec_params.local_work_size[1] = wg_size_local;
						// pad global work size to a multiple of the work group size.
						// this is necessary to make the local memory scheme work.
						exec_params.global_work_size[0] = ((exec_params.global_work_size[0] + wg_size_local - 1) / wg_size_local) * wg_size_local;
						exec_params.global_work_size[1] = ((exec_params.global_work_size[1] + wg_size_local - 1) / wg_size_local) * wg_size_local;
						// other arguments
						cl_int2 input_size{texture.response.cols(), texture.response.rows()};
						cl_int2 output_size{response_dims.width, response_dims.height};
						cl_int2 kernel_size{kernel.response.cols(), kernel.response.rows()};
						cl_int2 input_piv{rotated_kernel_overlaps[0], rotated_kernel_overlaps[2]};
						cl_float2 rotation_sincos{std::sinf(static_cast<float>(texture_rotation)), std::cosf(static_cast<float>(texture_rotation))};
						cl_int kernel_offset{0};
						cl_int num_kernel_pixels{kernel.response.cols() * kernel.response.rows()};
						// first pass
						simple_cl::cl::Event first_event{(*m_program_sqdiff_constant_local)(
							m_kernel_constant_sqdiff_local,
							pre_compute_events.begin(),
							pre_compute_events.end(),
							exec_params,
							*(input_image.images[0]),
							simple_cl::cl::LocalMemory<cl_float4>(local_buffer_total_size),
							*(m_kernel_buffer.buffer),
							*(m_output_buffer_a),
							input_size,
							output_size,
							kernel_size,
							cl_int2{kernel_anchor.x, kernel_anchor.y},
							input_piv,
							cl_int4{rotated_kernel_overlaps[0], rotated_kernel_overlaps[1], rotated_kernel_overlaps[2], rotated_kernel_overlaps[3]},
							rotation_sincos)
						};
						pre_compute_events.clear();
						pre_compute_events.push_back(std::move(first_event));
						// if necessary, more passes
						for(std::size_t batch{1}; batch < num_batches; ++batch) // ping pong between two output buffers
						{
							kernel_offset += num_kernel_pixels;
							if(batch % 2ull == 0)
							{
								simple_cl::cl::Event event{(*m_program_sqdiff_constant_local)(
									m_kernel_constant_sqdiff_local_nth_pass,
									pre_compute_events.begin(),
									pre_compute_events.end(),
									exec_params,
									*(input_image.images[batch]),
									simple_cl::cl::LocalMemory<cl_float4>(local_buffer_total_size),
									*(m_kernel_buffer.buffer),
									*(m_output_buffer_b),
									*(m_output_buffer_a),
									input_size,
									output_size,
									kernel_size,
									cl_int2{kernel_anchor.x, kernel_anchor.y},
									input_piv,
									cl_int4{rotated_kernel_overlaps[0], rotated_kernel_overlaps[1], rotated_kernel_overlaps[2], rotated_kernel_overlaps[3]},
									rotation_sincos,
									kernel_offset)
								};
								pre_compute_events.clear();
								pre_compute_events.push_back(std::move(event));
							}
							else
							{
								simple_cl::cl::Event event{(*m_program_sqdiff_constant_local)(
									m_kernel_constant_sqdiff_local_nth_pass,
									pre_compute_events.begin(),
									pre_compute_events.end(),
									exec_params,
									*(input_image.images[batch]),
									simple_cl::cl::LocalMemory<cl_float4>(local_buffer_total_size),
									*(m_kernel_buffer.buffer),
									*(m_output_buffer_a),
									*(m_output_buffer_b),
									input_size,
									output_size,
									kernel_size,
									cl_int2{kernel_anchor.x, kernel_anchor.y},
									input_piv,
									cl_int4{rotated_kernel_overlaps[0], rotated_kernel_overlaps[1], rotated_kernel_overlaps[2], rotated_kernel_overlaps[3]},
									rotation_sincos,
									kernel_offset)
								};
								pre_compute_events.clear();
								pre_compute_events.push_back(std::move(event));
							}
						}
					}
				}
				// prepare stuff for minimum extraction while kernel is running
				std::size_t find_min_local_work_size{get_local_work_size(m_kernel_find_min)};
				simple_cl::cl::Program::ExecParams find_min_exec_params{
					2ull,
					{0ull, 0ull, 0ull},
					{0ull, 0ull, 1ull},
					{find_min_local_work_size, find_min_local_work_size, 1ull}
				};
				std::size_t find_min_local_buffer_size;
				prepare_find_min_output_buffer(cv::Size{response_dims.width, response_dims.height}, find_min_local_work_size, find_min_exec_params.global_work_size[0], find_min_exec_params.global_work_size[1], find_min_local_buffer_size);

				// read result and wait for command chain to finish execution. If num_batches is odd, read output a, else b.
				simple_cl::cl::Event response_finished_event{read_output_image(match_res_out.total_cost_matrix, response_dims, pre_compute_events, num_batches % 2ull)};
				pre_compute_events.clear();
				pre_compute_events.push_back(std::move(response_finished_event));

				// run kernel for minimum extraction
				simple_cl::cl::Event find_min_kernel_event{(*m_program_find_min)(
					m_kernel_find_min,
					pre_compute_events.begin(),
					pre_compute_events.end(),
					find_min_exec_params,
					(num_batches % 2ull ? *m_output_buffer_a : *m_output_buffer_b),
					*(m_output_buffer_find_min.buffer),
					simple_cl::cl::LocalMemory<cl_float4>(find_min_local_buffer_size),
					cl_int2{response_dims.width, response_dims.height})
				};
				pre_compute_events.clear();
				pre_compute_events.push_back(std::move(find_min_kernel_event));
				// output result
				cv::Point result_offset = cv::Point(rotated_kernel_overlaps[0], rotated_kernel_overlaps[2]);
				read_min_pos_and_cost(match_res_out, pre_compute_events, result_offset);
			}

			inline void ocl_patch_matching::matching_policies::impl::CLMatcherImpl::compute_matches(
				const Texture& texture,
				const cv::Mat& texture_mask,
				const Texture& kernel,
				double texture_rotation,
				MatchingResult& match_res_out,
				bool erode_texture_mask)
			{
				static std::vector<simple_cl::cl::Event> pre_compute_events;
				pre_compute_events.clear();
				// compute rotated kernel size
				// kernel anchor
				cv::Point kernel_anchor{(m_result_origin == CLMatcher::ResultOrigin::Center ? cv::Point((kernel.response.cols() - 1) / 2, (kernel.response.rows() - 1) / 2) : cv::Point(0, 0))};
				cv::Size rotated_kernel_size;
				cv::Vec4i rotated_kernel_overlaps;
				calculate_rotated_kernel_dims(rotated_kernel_size, rotated_kernel_overlaps, kernel, texture_rotation, kernel_anchor);
				auto response_dims{get_response_dimensions(texture, kernel, texture_rotation, kernel_anchor)};
				// prepare all input data
				// upload input image
				prepare_input_image(texture, pre_compute_events, false, false);
				// upload kernel data
				bool use_constant{use_constant_kernel(kernel)};
				if(use_constant)
					prepare_kernel_buffer(kernel, pre_compute_events, false);
				else
					prepare_kernel_image(kernel, pre_compute_events, false);
				// output images
				prepare_output_image(texture, kernel, texture_rotation, response_dims);
				// get input image from map
				InputImage& input_image{m_input_images[m_texture_index_map[texture.id]]};
				// pingpong between the two output buffers until done
				std::size_t num_feature_maps{static_cast<std::size_t>(texture.response.num_channels())};
				std::size_t num_batches{num_feature_maps / 4ull + (num_feature_maps % 4ull != 0ull ? 1ull : 0ull)};
				// exec params				
				simple_cl::cl::Program::ExecParams exec_params{
					2ull,
					{0ull, 0ull, 0ull},
					{static_cast<std::size_t>(response_dims.width), static_cast<std::size_t>(response_dims.height), 1ull},
					{m_local_block_size, m_local_block_size, 1ull}
				};
				if(!use_constant)
				{
					std::size_t wg_size{std::min(get_local_work_size(m_kernel_naive_sqdiff), get_local_work_size(m_kernel_naive_sqdiff_nth_pass))};
					exec_params.local_work_size[0] = wg_size;
					exec_params.local_work_size[1] = wg_size;
					// other arguments
					cl_int2 input_size{texture.response.cols(), texture.response.rows()};
					cl_int2 kernel_size{kernel.response.cols(), kernel.response.rows()};
					cl_int2 input_piv{rotated_kernel_overlaps[0], rotated_kernel_overlaps[2]};
					cl_float2 rotation_sincos{std::sinf(static_cast<float>(texture_rotation)), std::cosf(static_cast<float>(texture_rotation))};
					// first pass
					simple_cl::cl::Event first_event{(*m_program_naive_sqdiff)(
						m_kernel_naive_sqdiff,
						pre_compute_events.begin(),
						pre_compute_events.end(),
						exec_params,
						*(input_image.images[0]),
						*(m_kernel_image.images[0]),
						*(m_output_buffer_a),
						input_size,
						kernel_size,
						cl_int2{kernel_anchor.x, kernel_anchor.y},
						input_piv,
						rotation_sincos)
					};
					pre_compute_events.clear();
					pre_compute_events.push_back(std::move(first_event));
					// if necessary, more passes
					for(std::size_t batch{1}; batch < num_batches; ++batch) // ping pong between two output buffers
					{
						if(batch % 2ull == 0)
						{
							simple_cl::cl::Event event{(*m_program_naive_sqdiff)(
								m_kernel_naive_sqdiff_nth_pass,
								pre_compute_events.begin(),
								pre_compute_events.end(),
								exec_params,
								*(input_image.images[batch]),
								*(m_kernel_image.images[batch]),
								*(m_output_buffer_b),
								*(m_output_buffer_a),
								input_size,
								kernel_size,
								cl_int2{kernel_anchor.x, kernel_anchor.y},
								input_piv,
								rotation_sincos)
							};
							pre_compute_events.clear();
							pre_compute_events.push_back(std::move(event));
						}
						else
						{
							simple_cl::cl::Event event{(*m_program_naive_sqdiff)(
								m_kernel_naive_sqdiff_nth_pass,
								pre_compute_events.begin(),
								pre_compute_events.end(),
								exec_params,
								*(input_image.images[batch]),
								*(m_kernel_image.images[batch]),
								*(m_output_buffer_a),
								*(m_output_buffer_b),
								input_size,
								kernel_size,
								cl_int2{kernel_anchor.x, kernel_anchor.y},
								input_piv,
								rotation_sincos)
							};
							pre_compute_events.clear();
							pre_compute_events.push_back(std::move(event));
						}
					}
				}
				else
				{
					// get safe local wg size
					std::size_t wg_size{std::min(get_local_work_size(m_kernel_constant_sqdiff), get_local_work_size(m_kernel_constant_sqdiff_nth_pass))};
					std::size_t wg_size_local{std::min(get_local_work_size(m_kernel_constant_sqdiff_local), get_local_work_size(m_kernel_constant_sqdiff_local_nth_pass))};
					std::size_t wg_used_local_mem{std::max(m_kernel_constant_sqdiff_local.getKernelInfo().local_memory_usage, m_kernel_constant_sqdiff_local_nth_pass.getKernelInfo().local_memory_usage)};

					// calculate total buffer size in pixels
					std::size_t local_buffer_total_size{static_cast<std::size_t>(rotated_kernel_overlaps[0] + wg_size_local + rotated_kernel_overlaps[1]) * static_cast<std::size_t>(rotated_kernel_overlaps[2] + wg_size_local + rotated_kernel_overlaps[3])};

					bool use_local{use_local_mem(rotated_kernel_overlaps, wg_used_local_mem, wg_size_local, m_local_buffer_max_pixels, sizeof(cl_float4)) && m_use_local_buffer_for_matching};

					if(!use_local)
					{
						exec_params.local_work_size[0] = wg_size;
						exec_params.local_work_size[1] = wg_size;
						// other arguments
						cl_int2 input_size{texture.response.cols(), texture.response.rows()};
						cl_int2 kernel_size{kernel.response.cols(), kernel.response.rows()};
						cl_int2 input_piv{rotated_kernel_overlaps[0], rotated_kernel_overlaps[2]};
						cl_float2 rotation_sincos{std::sinf(static_cast<float>(texture_rotation)), std::cosf(static_cast<float>(texture_rotation))};
						cl_int kernel_offset{0};
						cl_int num_kernel_pixels{kernel.response.cols() * kernel.response.rows()};
						// first pass
						simple_cl::cl::Event first_event{(*m_program_sqdiff_constant)(
							m_kernel_constant_sqdiff,
							pre_compute_events.begin(),
							pre_compute_events.end(),
							exec_params,
							*(input_image.images[0]),
							*(m_kernel_buffer.buffer),
							*(m_output_buffer_a),
							input_size,
							kernel_size,
							cl_int2{kernel_anchor.x, kernel_anchor.y},
							input_piv,
							rotation_sincos)
						};
						pre_compute_events.clear();
						pre_compute_events.push_back(std::move(first_event));
						// if necessary, more passes
						for(std::size_t batch{1}; batch < num_batches; ++batch) // ping pong between two output buffers
						{
							kernel_offset += num_kernel_pixels;
							if(batch % 2ull == 0)
							{
								simple_cl::cl::Event event{(*m_program_sqdiff_constant)(
									m_kernel_constant_sqdiff_nth_pass,
									pre_compute_events.begin(),
									pre_compute_events.end(),
									exec_params,
									*(input_image.images[batch]),
									*(m_kernel_buffer.buffer),
									*(m_output_buffer_b),
									*(m_output_buffer_a),
									input_size,
									kernel_size,
									cl_int2{kernel_anchor.x, kernel_anchor.y},
									input_piv,
									rotation_sincos,
									kernel_offset)
								};
								pre_compute_events.clear();
								pre_compute_events.push_back(std::move(event));
							}
							else
							{
								simple_cl::cl::Event event{(*m_program_sqdiff_constant)(
									m_kernel_constant_sqdiff_nth_pass,
									pre_compute_events.begin(),
									pre_compute_events.end(),
									exec_params,
									*(input_image.images[batch]),
									*(m_kernel_buffer.buffer),
									*(m_output_buffer_a),
									*(m_output_buffer_b),
									input_size,
									kernel_size,
									cl_int2{kernel_anchor.x, kernel_anchor.y},
									input_piv,
									rotation_sincos,
									kernel_offset)
								};
								pre_compute_events.clear();
								pre_compute_events.push_back(std::move(event));
							}
						}
					}
					else
					{
						exec_params.local_work_size[0] = wg_size_local;
						exec_params.local_work_size[1] = wg_size_local;
						// pad global work size to a multiple of the work group size.
						// this is necessary to make the local memory scheme work.
						exec_params.global_work_size[0] = ((exec_params.global_work_size[0] + wg_size_local - 1) / wg_size_local) * wg_size_local;
						exec_params.global_work_size[1] = ((exec_params.global_work_size[1] + wg_size_local - 1) / wg_size_local) * wg_size_local;
						// other arguments
						cl_int2 input_size{texture.response.cols(), texture.response.rows()};
						cl_int2 output_size{response_dims.width, response_dims.height};
						cl_int2 kernel_size{kernel.response.cols(), kernel.response.rows()};
						cl_int2 input_piv{rotated_kernel_overlaps[0], rotated_kernel_overlaps[2]};
						cl_float2 rotation_sincos{std::sinf(static_cast<float>(texture_rotation)), std::cosf(static_cast<float>(texture_rotation))};
						cl_int kernel_offset{0};
						cl_int num_kernel_pixels{kernel.response.cols() * kernel.response.rows()};
						// first pass
						simple_cl::cl::Event first_event{(*m_program_sqdiff_constant_local)(
							m_kernel_constant_sqdiff_local,
							pre_compute_events.begin(),
							pre_compute_events.end(),
							exec_params,
							*(input_image.images[0]),
							simple_cl::cl::LocalMemory<cl_float4>(local_buffer_total_size),
							*(m_kernel_buffer.buffer),
							*(m_output_buffer_a),
							input_size,
							output_size,
							kernel_size,
							cl_int2{kernel_anchor.x, kernel_anchor.y},
							input_piv,
							cl_int4{rotated_kernel_overlaps[0], rotated_kernel_overlaps[1], rotated_kernel_overlaps[2], rotated_kernel_overlaps[3]},
							rotation_sincos)
						};
						pre_compute_events.clear();
						pre_compute_events.push_back(std::move(first_event));
						// if necessary, more passes
						for(std::size_t batch{1}; batch < num_batches; ++batch) // ping pong between two output buffers
						{
							kernel_offset += num_kernel_pixels;
							if(batch % 2ull == 0)
							{
								simple_cl::cl::Event event{(*m_program_sqdiff_constant_local)(
									m_kernel_constant_sqdiff_local_nth_pass,
									pre_compute_events.begin(),
									pre_compute_events.end(),
									exec_params,
									*(input_image.images[batch]),
									simple_cl::cl::LocalMemory<cl_float4>(local_buffer_total_size),
									*(m_kernel_buffer.buffer),
									*(m_output_buffer_b),
									*(m_output_buffer_a),
									input_size,
									output_size,
									kernel_size,
									cl_int2{kernel_anchor.x, kernel_anchor.y},
									input_piv,
									cl_int4{rotated_kernel_overlaps[0], rotated_kernel_overlaps[1], rotated_kernel_overlaps[2], rotated_kernel_overlaps[3]},
									rotation_sincos,
									kernel_offset)
								};
								pre_compute_events.clear();
								pre_compute_events.push_back(std::move(event));
							}
							else
							{
								simple_cl::cl::Event event{(*m_program_sqdiff_constant_local)(
									m_kernel_constant_sqdiff_local_nth_pass,
									pre_compute_events.begin(),
									pre_compute_events.end(),
									exec_params,
									*(input_image.images[batch]),
									simple_cl::cl::LocalMemory<cl_float4>(local_buffer_total_size),
									*(m_kernel_buffer.buffer),
									*(m_output_buffer_a),
									*(m_output_buffer_b),
									input_size,
									output_size,
									kernel_size,
									cl_int2{kernel_anchor.x, kernel_anchor.y},
									input_piv,
									cl_int4{rotated_kernel_overlaps[0], rotated_kernel_overlaps[1], rotated_kernel_overlaps[2], rotated_kernel_overlaps[3]},
									rotation_sincos,
									kernel_offset)
								};
								pre_compute_events.clear();
								pre_compute_events.push_back(std::move(event));
							}
						}
					}
				}
				// prepare stuff for minimum extraction while kernel is running
				std::size_t find_min_local_work_size{get_local_work_size(m_kernel_find_min)};
				simple_cl::cl::Program::ExecParams find_min_exec_params{
					2ull,
					{0ull, 0ull, 0ull},
					{0ull, 0ull, 1ull},
					{find_min_local_work_size, find_min_local_work_size, 1ull}
				};
				std::size_t find_min_local_buffer_size;
				prepare_find_min_output_buffer(cv::Size{response_dims.width, response_dims.height}, find_min_local_work_size, find_min_exec_params.global_work_size[0], find_min_exec_params.global_work_size[1], find_min_local_buffer_size);

				// texture mask
				static std::vector<simple_cl::cl::Event> texture_mask_events;
				texture_mask_events.clear();
				prepare_texture_mask(texture_mask, texture_mask_events, false);

				if(erode_texture_mask)
				{
					std::size_t wg_size{std::min(get_local_work_size(m_kernel_erode), get_local_work_size(m_kernel_erode_local))};
					std::size_t wg_size_local{std::min(get_local_work_size(m_kernel_erode), get_local_work_size(m_kernel_erode_local))};
					std::size_t wg_used_local_mem{std::max(m_kernel_erode.getKernelInfo().local_memory_usage, m_kernel_erode_local.getKernelInfo().local_memory_usage)};
					bool erode_use_local{use_local_mem(rotated_kernel_overlaps, m_kernel_erode_local.getKernelInfo().local_memory_usage, wg_size_local, m_local_buffer_max_pixels * 4ull, sizeof(cl_float)) && m_use_local_buffer_for_erode};
					// erode texture mask with kernel mask				
					prepare_erode_output_image(texture_mask);					
					if(!erode_use_local)
					{
						// execution params for kernel
						simple_cl::cl::Program::ExecParams erode_exec_params{
							2ull,
							{0ull, 0ull, 0ull},
							{static_cast<std::size_t>(texture_mask.cols), static_cast<std::size_t>(texture_mask.rows), 1ull},
							{wg_size, wg_size, 1ull}
						};
						simple_cl::cl::Event event{(*m_program_erode)(m_kernel_erode,
							texture_mask_events.begin(),
							texture_mask_events.end(),
							erode_exec_params,
							*m_texture_mask,
							*m_output_texture_mask_eroded,
							cl_int2{texture_mask.cols, texture_mask.rows},
							cl_int2{kernel.response.cols(), kernel.response.rows()},
							cl_int2{kernel_anchor.x, kernel_anchor.y},
							cl_float2{std::sinf(static_cast<float>(texture_rotation)), std::cosf(static_cast<float>(texture_rotation))}
						)};
						texture_mask_events.clear();
						texture_mask_events.push_back(event);
					}
					else
					{
						// calculate total buffer size in pixels
						std::size_t erode_local_buffer_total_size{static_cast<std::size_t>(rotated_kernel_overlaps[0] + wg_size_local + rotated_kernel_overlaps[1]) * static_cast<std::size_t>(rotated_kernel_overlaps[2] + wg_size_local + rotated_kernel_overlaps[3])};
						// execution params for kernel
						simple_cl::cl::Program::ExecParams erode_exec_params{
							2ull,
							{0ull, 0ull, 0ull},
							{static_cast<std::size_t>(texture_mask.cols), static_cast<std::size_t>(texture_mask.rows), 1ull},
							{wg_size_local, wg_size_local, 1ull}
						};
						erode_exec_params.global_work_size[0] = ((erode_exec_params.global_work_size[0] + wg_size_local - 1) / wg_size_local) * wg_size_local;
						erode_exec_params.global_work_size[1] = ((erode_exec_params.global_work_size[1] + wg_size_local - 1) / wg_size_local) * wg_size_local;
						simple_cl::cl::Event event{(*m_program_erode_local)(m_kernel_erode_local,
							texture_mask_events.begin(),
							texture_mask_events.end(),
							erode_exec_params,
							*m_texture_mask,
							*m_output_texture_mask_eroded,
							simple_cl::cl::LocalMemory<cl_float>(erode_local_buffer_total_size),
							cl_int2{texture_mask.cols, texture_mask.rows},
							cl_int2{texture_mask.cols, texture_mask.rows},
							cl_int2{kernel.response.cols(), kernel.response.rows()},
							cl_int2{kernel_anchor.x, kernel_anchor.y},
							cl_int4{rotated_kernel_overlaps[0], rotated_kernel_overlaps[1], rotated_kernel_overlaps[2], rotated_kernel_overlaps[3]},
							cl_float2{std::sinf(static_cast<float>(texture_rotation)), std::cosf(static_cast<float>(texture_rotation))}
						)};
						texture_mask_events.clear();
						texture_mask_events.push_back(event);
					}
				}

				// read result and wait for command chain to finish execution. If num_batches is odd, read output a, else b.
				simple_cl::cl::Event response_finished_event{read_output_image(match_res_out.total_cost_matrix, response_dims, pre_compute_events, num_batches % 2ull)};
				pre_compute_events.clear();
				pre_compute_events.push_back(std::move(response_finished_event));
				pre_compute_events.insert(pre_compute_events.end(), texture_mask_events.begin(), texture_mask_events.end());

				// run kernel for minimum extraction
				simple_cl::cl::Event find_min_kernel_event{(*m_program_find_min)(
					m_kernel_find_min_masked,
					pre_compute_events.begin(),
					pre_compute_events.end(),
					find_min_exec_params,
					(num_batches % 2ull ? *m_output_buffer_a : *m_output_buffer_b),
					(erode_texture_mask ? *m_output_texture_mask_eroded : *m_texture_mask),
					*(m_output_buffer_find_min.buffer),
					simple_cl::cl::LocalMemory<cl_float4>(find_min_local_buffer_size),
					cl_int2{response_dims.width, response_dims.height},
					cl_int2{rotated_kernel_overlaps[0], rotated_kernel_overlaps[2]})
				};
				pre_compute_events.clear();
				pre_compute_events.push_back(std::move(find_min_kernel_event));
				// output result
				cv::Point result_offset = cv::Point(rotated_kernel_overlaps[0], rotated_kernel_overlaps[2]);
				read_min_pos_and_cost(match_res_out, pre_compute_events, result_offset);
			}

			inline void ocl_patch_matching::matching_policies::impl::CLMatcherImpl::compute_matches(
				const Texture& texture,
				const cv::Mat& texture_mask,
				const Texture& kernel,
				const cv::Mat& kernel_mask,
				double texture_rotation,
				MatchingResult& match_res_out,
				bool erode_texture_mask)
			{
				static std::vector<simple_cl::cl::Event> pre_compute_events;
				pre_compute_events.clear();
				// compute rotated kernel size
				// kernel anchor
				cv::Point kernel_anchor{(m_result_origin == CLMatcher::ResultOrigin::Center ? cv::Point((kernel.response.cols() - 1) / 2, (kernel.response.rows() - 1) / 2) : cv::Point(0, 0))};
				cv::Size rotated_kernel_size;
				cv::Vec4i rotated_kernel_overlaps;
				calculate_rotated_kernel_dims(rotated_kernel_size, rotated_kernel_overlaps, kernel, texture_rotation, kernel_anchor);
				auto response_dims{get_response_dimensions(texture, kernel, texture_rotation, kernel_anchor)};
				// prepare all input data
				prepare_input_image(texture, pre_compute_events, false, false);
				bool use_constant{use_constant_kernel(kernel, kernel_mask)};
				if(use_constant)
				{
					prepare_kernel_buffer(kernel, pre_compute_events, false);
					prepare_kernel_mask_buffer(kernel_mask, pre_compute_events, false);
				}
				else
				{
					prepare_kernel_image(kernel, pre_compute_events, false);
					prepare_kernel_mask(kernel_mask, pre_compute_events, false);
				}
				prepare_output_image(texture, kernel, texture_rotation, response_dims);
				// get input image from map
				InputImage& input_image{m_input_images[m_texture_index_map[texture.id]]};
				// pingpong between the two output buffers until done
				std::size_t num_feature_maps{static_cast<std::size_t>(texture.response.num_channels())};
				std::size_t num_batches{num_feature_maps / 4ull + (num_feature_maps % 4ull != 0ull ? 1ull : 0ull)};
				// exec params
				simple_cl::cl::Program::ExecParams exec_params{
					2ull,
					{0ull, 0ull, 0ull},
					{static_cast<std::size_t>(response_dims.width), static_cast<std::size_t>(response_dims.height), 1ull},
					{m_local_block_size, m_local_block_size, 1ull}
				};
				if(!use_constant)
				{
					std::size_t wg_size{std::min(get_local_work_size(m_kernel_naive_sqdiff_masked), get_local_work_size(m_kernel_naive_sqdiff_masked_nth_pass))};
					exec_params.local_work_size[0] = wg_size;
					exec_params.local_work_size[1] = wg_size;
					// other arguments
					cl_int2 input_size{texture.response.cols(), texture.response.rows()};
					cl_int2 kernel_size{kernel.response.cols(), kernel.response.rows()};
					cl_int2 input_piv{rotated_kernel_overlaps[0], rotated_kernel_overlaps[2]};
					cl_float2 rotation_sincos{std::sinf(static_cast<float>(texture_rotation)), std::cosf(static_cast<float>(texture_rotation))};
					// first pass
					simple_cl::cl::Event first_event{(*m_program_naive_sqdiff)(
						m_kernel_naive_sqdiff_masked,
						pre_compute_events.begin(),
						pre_compute_events.end(),
						exec_params,
						*(input_image.images[0]),
						*(m_kernel_image.images[0]),
						*(m_kernel_mask),
						*(m_output_buffer_a),
						input_size,
						kernel_size,
						cl_int2{kernel_anchor.x, kernel_anchor.y},
						input_piv,
						rotation_sincos)
					};
					pre_compute_events.clear();
					pre_compute_events.push_back(std::move(first_event));
					// if necessary, more passes
					for(std::size_t batch{1}; batch < num_batches; ++batch) // ping pong between two output buffers
					{
						if(batch % 2ull == 0)
						{
							simple_cl::cl::Event event{(*m_program_naive_sqdiff)(
								m_kernel_naive_sqdiff_masked_nth_pass,
								pre_compute_events.begin(),
								pre_compute_events.end(),
								exec_params,
								*(input_image.images[batch]),
								*(m_kernel_image.images[batch]),
								*(m_kernel_mask),
								*(m_output_buffer_b),
								*(m_output_buffer_a),
								input_size,
								kernel_size,
								cl_int2{kernel_anchor.x, kernel_anchor.y},
								input_piv,
								rotation_sincos)
							};
							pre_compute_events.clear();
							pre_compute_events.push_back(std::move(event));
						}
						else
						{
							simple_cl::cl::Event event{(*m_program_naive_sqdiff)(
								m_kernel_naive_sqdiff_masked_nth_pass,
								pre_compute_events.begin(),
								pre_compute_events.end(),
								exec_params,
								*(input_image.images[batch]),
								*(m_kernel_image.images[batch]),
								*(m_kernel_mask),
								*(m_output_buffer_a),
								*(m_output_buffer_b),
								input_size,
								kernel_size,
								cl_int2{kernel_anchor.x, kernel_anchor.y},
								input_piv,
								rotation_sincos)
							};
							pre_compute_events.clear();
							pre_compute_events.push_back(std::move(event));
						}
					}
				}
				else
				{					
					// get safe local wg size
					std::size_t wg_size{std::min(get_local_work_size(m_kernel_constant_sqdiff_masked), get_local_work_size(m_kernel_constant_sqdiff_masked_nth_pass))};
					std::size_t wg_size_local{std::min(get_local_work_size(m_kernel_constant_sqdiff_local_masked), get_local_work_size(m_kernel_constant_sqdiff_local_masked_nth_pass))};
					std::size_t wg_used_local_mem{std::max(m_kernel_constant_sqdiff_local_masked.getKernelInfo().local_memory_usage, m_kernel_constant_sqdiff_local_masked_nth_pass.getKernelInfo().local_memory_usage)};

					// calculate total buffer size in pixels
					std::size_t local_buffer_total_size{static_cast<std::size_t>(rotated_kernel_overlaps[0] + wg_size_local + rotated_kernel_overlaps[1]) * static_cast<std::size_t>(rotated_kernel_overlaps[2] + wg_size_local + rotated_kernel_overlaps[3])};

					// decide if we should use local memory optimization
					bool use_local{use_local_mem(rotated_kernel_overlaps, wg_used_local_mem, wg_size_local, m_local_buffer_max_pixels, sizeof(cl_float4)) && m_use_local_buffer_for_matching};

					if(!use_local)
					{
						exec_params.local_work_size[0] = wg_size;
						exec_params.local_work_size[1] = wg_size;
						// other arguments
						cl_int2 input_size{texture.response.cols(), texture.response.rows()};
						cl_int2 kernel_size{kernel.response.cols(), kernel.response.rows()};
						cl_int2 input_piv{rotated_kernel_overlaps[0], rotated_kernel_overlaps[2]};
						cl_float2 rotation_sincos{std::sinf(static_cast<float>(texture_rotation)), std::cosf(static_cast<float>(texture_rotation))};
						cl_int kernel_offset{0};
						cl_int num_kernel_pixels{kernel.response.cols() * kernel.response.rows()};
						// first pass
						simple_cl::cl::Event first_event{(*m_program_sqdiff_constant)(
							m_kernel_constant_sqdiff_masked,
							pre_compute_events.begin(),
							pre_compute_events.end(),
							exec_params,
							*(input_image.images[0]),
							*(m_kernel_buffer.buffer),
							*(m_kernel_mask_buffer.buffer),
							*(m_output_buffer_a),
							input_size,
							kernel_size,
							cl_int2{kernel_anchor.x, kernel_anchor.y},
							input_piv,
							rotation_sincos)
						};
						pre_compute_events.clear();
						pre_compute_events.push_back(std::move(first_event));
						// if necessary, more passes
						for(std::size_t batch{1}; batch < num_batches; ++batch) // ping pong between two output buffers
						{
							kernel_offset += num_kernel_pixels;
							if(batch % 2ull == 0)
							{
								simple_cl::cl::Event event{(*m_program_sqdiff_constant)(
									m_kernel_constant_sqdiff_masked_nth_pass,
									pre_compute_events.begin(),
									pre_compute_events.end(),
									exec_params,
									*(input_image.images[batch]),
									*(m_kernel_buffer.buffer),
									*(m_kernel_mask_buffer.buffer),
									*(m_output_buffer_b),
									*(m_output_buffer_a),
									input_size,
									kernel_size,
									cl_int2{kernel_anchor.x, kernel_anchor.y},
									input_piv,
									rotation_sincos,
									kernel_offset)
								};
								pre_compute_events.clear();
								pre_compute_events.push_back(std::move(event));
							}
							else
							{
								simple_cl::cl::Event event{(*m_program_sqdiff_constant)(
									m_kernel_constant_sqdiff_masked_nth_pass,
									pre_compute_events.begin(),
									pre_compute_events.end(),
									exec_params,
									*(input_image.images[batch]),
									*(m_kernel_buffer.buffer),
									*(m_kernel_mask_buffer.buffer),
									*(m_output_buffer_a),
									*(m_output_buffer_b),
									input_size,
									kernel_size,
									cl_int2{kernel_anchor.x, kernel_anchor.y},
									input_piv,
									rotation_sincos,
									kernel_offset)
								};
								pre_compute_events.clear();
								pre_compute_events.push_back(std::move(event));
							}
						}
					}
					else
					{
						exec_params.local_work_size[0] = wg_size_local;
						exec_params.local_work_size[1] = wg_size_local;
						// pad global work size to a multiple of the work group size.
						// this is necessary to make the local memory scheme work.
						exec_params.global_work_size[0] = ((exec_params.global_work_size[0] + wg_size_local - 1) / wg_size_local) * wg_size_local;
						exec_params.global_work_size[1] = ((exec_params.global_work_size[1] + wg_size_local - 1) / wg_size_local) * wg_size_local;
						// other arguments
						cl_int2 input_size{texture.response.cols(), texture.response.rows()};
						cl_int2 output_size{response_dims.width, response_dims.height};
						cl_int2 kernel_size{kernel.response.cols(), kernel.response.rows()};
						cl_int2 input_piv{rotated_kernel_overlaps[0], rotated_kernel_overlaps[2]};
						cl_float2 rotation_sincos{std::sinf(static_cast<float>(texture_rotation)), std::cosf(static_cast<float>(texture_rotation))};
						cl_int kernel_offset{0};
						cl_int num_kernel_pixels{kernel.response.cols() * kernel.response.rows()};
						// first pass
						simple_cl::cl::Event first_event{(*m_program_sqdiff_constant_local_masked)(
							m_kernel_constant_sqdiff_local_masked,
							pre_compute_events.begin(),
							pre_compute_events.end(),
							exec_params,
							*(input_image.images[0]),
							simple_cl::cl::LocalMemory<cl_float4>(local_buffer_total_size),
							*(m_kernel_buffer.buffer),
							*(m_kernel_mask_buffer.buffer),
							*(m_output_buffer_a),
							input_size,
							output_size,
							kernel_size,
							cl_int2{kernel_anchor.x, kernel_anchor.y},
							input_piv,
							cl_int4{rotated_kernel_overlaps[0], rotated_kernel_overlaps[1], rotated_kernel_overlaps[2], rotated_kernel_overlaps[3]},
							rotation_sincos)
						};
						pre_compute_events.clear();
						pre_compute_events.push_back(std::move(first_event));
						// if necessary, more passes
						for(std::size_t batch{1}; batch < num_batches; ++batch) // ping pong between two output buffers
						{
							kernel_offset += num_kernel_pixels;
							if(batch % 2ull == 0)
							{
								simple_cl::cl::Event event{(*m_program_sqdiff_constant_local_masked)(
									m_kernel_constant_sqdiff_local_masked_nth_pass,
									pre_compute_events.begin(),
									pre_compute_events.end(),
									exec_params,
									*(input_image.images[batch]),
									simple_cl::cl::LocalMemory<cl_float4>(local_buffer_total_size),
									*(m_kernel_buffer.buffer),
									*(m_kernel_mask_buffer.buffer),
									*(m_output_buffer_b),
									*(m_output_buffer_a),
									input_size,
									output_size,
									kernel_size,
									cl_int2{kernel_anchor.x, kernel_anchor.y},
									input_piv,
									cl_int4{rotated_kernel_overlaps[0], rotated_kernel_overlaps[1], rotated_kernel_overlaps[2], rotated_kernel_overlaps[3]},
									rotation_sincos,
									kernel_offset)
								};
								pre_compute_events.clear();
								pre_compute_events.push_back(std::move(event));
							}
							else
							{
								simple_cl::cl::Event event{(*m_program_sqdiff_constant_local_masked)(
									m_kernel_constant_sqdiff_local_masked_nth_pass,
									pre_compute_events.begin(),
									pre_compute_events.end(),
									exec_params,
									*(input_image.images[batch]),
									simple_cl::cl::LocalMemory<cl_float4>(local_buffer_total_size),
									*(m_kernel_buffer.buffer),
									*(m_kernel_mask_buffer.buffer),
									*(m_output_buffer_a),
									*(m_output_buffer_b),
									input_size,
									output_size,
									kernel_size,
									cl_int2{kernel_anchor.x, kernel_anchor.y},
									input_piv,
									cl_int4{rotated_kernel_overlaps[0], rotated_kernel_overlaps[1], rotated_kernel_overlaps[2], rotated_kernel_overlaps[3]},
									rotation_sincos,
									kernel_offset)
								};
								pre_compute_events.clear();
								pre_compute_events.push_back(std::move(event));
							}
						}
					}
				}
				// prepare stuff for minimum extraction while kernel is running
				std::size_t find_min_local_work_size{get_local_work_size(m_kernel_find_min)};
				simple_cl::cl::Program::ExecParams find_min_exec_params{
					2ull,
					{0ull, 0ull, 0ull},
					{0ull, 0ull, 1ull},
					{find_min_local_work_size, find_min_local_work_size, 1ull}
				};
				std::size_t find_min_local_buffer_size;
				prepare_find_min_output_buffer(cv::Size{response_dims.width, response_dims.height}, find_min_local_work_size, find_min_exec_params.global_work_size[0], find_min_exec_params.global_work_size[1], find_min_local_buffer_size);

				// texture mask
				static std::vector<simple_cl::cl::Event> texture_mask_events;
				texture_mask_events.clear();
				prepare_texture_mask(texture_mask, texture_mask_events, false);

				cv::Mat texmask;
				if(erode_texture_mask)
				{
					// erode texture mask with kernel mask				
					prepare_erode_output_image(texture_mask);

					if(use_constant_kernel(kernel_mask))
					{
						std::size_t wg_size{std::min(get_local_work_size(m_kernel_erode_masked), get_local_work_size(m_kernel_erode_masked_local))};
						std::size_t wg_size_local{std::min(get_local_work_size(m_kernel_erode_masked), get_local_work_size(m_kernel_erode_masked_local))};
						std::size_t wg_used_local_mem{std::max(m_kernel_erode_masked.getKernelInfo().local_memory_usage, m_kernel_erode_masked_local.getKernelInfo().local_memory_usage)};
						bool erode_use_local{use_local_mem(rotated_kernel_overlaps, m_kernel_erode_masked_local.getKernelInfo().local_memory_usage, wg_size_local, m_local_buffer_max_pixels * 4ull, sizeof(cl_float)) && m_use_local_buffer_for_erode};
						if(!erode_use_local)
						{
							simple_cl::cl::Program::ExecParams erode_exec_params{
								2ull,
								{0ull, 0ull, 0ull},
								{static_cast<std::size_t>(texture_mask.cols), static_cast<std::size_t>(texture_mask.rows), 1ull},
								{wg_size, wg_size, 1ull}
							};
							simple_cl::cl::Event event{(*m_program_erode_masked)(m_kernel_erode_constant_masked,
								texture_mask_events.begin(),
								texture_mask_events.end(),
								erode_exec_params,
								*m_texture_mask,
								*(m_kernel_mask_buffer.buffer),
								*m_output_texture_mask_eroded,
								cl_int2{texture_mask.cols, texture_mask.rows},
								cl_int2{kernel_mask.cols, kernel_mask.rows},
								cl_int2{kernel_anchor.x, kernel_anchor.y},
								cl_float2{std::sinf(static_cast<float>(texture_rotation)), std::cosf(static_cast<float>(texture_rotation))}
							)};
							texture_mask_events.clear();
							texture_mask_events.push_back(event);
						}
						else
						{
							simple_cl::cl::Program::ExecParams erode_exec_params{
								2ull,
								{0ull, 0ull, 0ull},
								{static_cast<std::size_t>(texture_mask.cols), static_cast<std::size_t>(texture_mask.rows), 1ull},
								{wg_size_local, wg_size_local, 1ull}
							};
							// calculate total buffer size in pixels
							std::size_t erode_local_buffer_total_size{static_cast<std::size_t>(rotated_kernel_overlaps[0] + wg_size_local + rotated_kernel_overlaps[1]) * static_cast<std::size_t>(rotated_kernel_overlaps[2] + wg_size_local + rotated_kernel_overlaps[3])};
							erode_exec_params.global_work_size[0] = ((erode_exec_params.global_work_size[0] + wg_size_local - 1) / wg_size_local) * wg_size_local;
							erode_exec_params.global_work_size[1] = ((erode_exec_params.global_work_size[1] + wg_size_local - 1) / wg_size_local) * wg_size_local;
							simple_cl::cl::Event event{(*m_program_erode_masked_local)(m_kernel_erode_masked_local,
								texture_mask_events.begin(),
								texture_mask_events.end(),
								erode_exec_params,
								*m_texture_mask,
								*(m_kernel_mask_buffer.buffer),
								*m_output_texture_mask_eroded,
								simple_cl::cl::LocalMemory<cl_float>(erode_local_buffer_total_size),
								cl_int2{texture_mask.cols, texture_mask.rows},
								cl_int2{texture_mask.cols, texture_mask.rows},
								cl_int2{kernel_mask.cols, kernel_mask.rows},
								cl_int2{kernel_anchor.x, kernel_anchor.y},
								cl_int4{rotated_kernel_overlaps[0], rotated_kernel_overlaps[1], rotated_kernel_overlaps[2], rotated_kernel_overlaps[3]},
								cl_float2{std::sinf(static_cast<float>(texture_rotation)), std::cosf(static_cast<float>(texture_rotation))}
							)};
							texture_mask_events.clear();
							texture_mask_events.push_back(event);
						}
					}
					else
					{
						simple_cl::cl::Program::ExecParams erode_exec_params{
								2ull,
								{0ull, 0ull, 0ull},
								{static_cast<std::size_t>(texture_mask.cols), static_cast<std::size_t>(texture_mask.rows), 1ull},
								{0ull, 0ull, 1ull}
						};
						std::size_t erode_local_work_size{get_local_work_size(m_kernel_erode_masked)};
						erode_exec_params.local_work_size[0] = erode_local_work_size;
						erode_exec_params.local_work_size[1] = erode_local_work_size;
						simple_cl::cl::Event event{(*m_program_erode_masked)(m_kernel_erode_masked,
							texture_mask_events.begin(),
							texture_mask_events.end(),
							exec_params,
							*m_texture_mask,
							*(m_kernel_mask),
							*m_output_texture_mask_eroded,
							cl_int2{texture_mask.cols, texture_mask.rows},
							cl_int2{kernel_mask.cols, kernel_mask.rows},
							cl_int2{kernel_anchor.x, kernel_anchor.y},
							cl_float2{std::sinf(static_cast<float>(texture_rotation)), std::cosf(static_cast<float>(texture_rotation))}
						)};
						texture_mask_events.clear();
						texture_mask_events.push_back(event);
					}
				}

				// read result and wait for command chain to finish execution. If num_batches is odd, read output a, else b.
				simple_cl::cl::Event response_finished_event{read_output_image(match_res_out.total_cost_matrix, response_dims, pre_compute_events, num_batches % 2ull)};
				pre_compute_events.clear();
				pre_compute_events.push_back(std::move(response_finished_event));
				pre_compute_events.insert(pre_compute_events.end(), texture_mask_events.begin(), texture_mask_events.end());

				// run kernel for minimum extraction
				simple_cl::cl::Event find_min_kernel_event{(*m_program_find_min)(
					m_kernel_find_min_masked,
					pre_compute_events.begin(),
					pre_compute_events.end(),
					find_min_exec_params,
					(num_batches % 2ull ? *m_output_buffer_a : *m_output_buffer_b),
					(erode_texture_mask ? *m_output_texture_mask_eroded : *m_texture_mask),
					*(m_output_buffer_find_min.buffer),
					simple_cl::cl::LocalMemory<cl_float4>(find_min_local_buffer_size),
					cl_int2{response_dims.width, response_dims.height},
					cl_int2{rotated_kernel_overlaps[0], rotated_kernel_overlaps[2]})
				};
				pre_compute_events.clear();
				pre_compute_events.push_back(std::move(find_min_kernel_event));
				// output result
				cv::Point result_offset = cv::Point(rotated_kernel_overlaps[0], rotated_kernel_overlaps[2]);
				read_min_pos_and_cost(match_res_out, pre_compute_events, result_offset);
			}

			inline cv::Vec3i ocl_patch_matching::matching_policies::impl::CLMatcherImpl::response_dimensions(
				const Texture& texture,
				const Texture& kernel,
				double texture_rotation) const
			{
				auto rdim{get_response_dimensions(texture, kernel, texture_rotation, m_result_origin == CLMatcher::ResultOrigin::Center ? cv::Point((kernel.response.cols() - 1) / 2, (kernel.response.rows() - 1) / 2) : cv::Point(0, 0))};
				return cv::Vec3i(rdim.width, rdim.height, 1);
			}

			inline match_response_cv_mat_t ocl_patch_matching::matching_policies::impl::CLMatcherImpl::response_image_data_type(
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
ocl_patch_matching::matching_policies::CLMatcher::CLMatcher(DeviceSelectionPolicy device_selection_policy, std::size_t max_texture_cache_memory, std::size_t local_block_size, std::size_t constant_kernel_max_pixels, std::size_t local_buffer_max_pixels, ResultOrigin result_origin, bool use_local_mem_for_matching, bool use_local_mem_for_erode) :
	m_impl(new impl::CLMatcherImpl(device_selection_policy, max_texture_cache_memory, local_block_size, constant_kernel_max_pixels, local_buffer_max_pixels, result_origin, use_local_mem_for_matching, use_local_mem_for_erode))
{
}

ocl_patch_matching::matching_policies::CLMatcher::~CLMatcher() noexcept
{
}

std::size_t ocl_patch_matching::matching_policies::CLMatcher::platform_id() const
{
	return impl()->platform_id();
}

std::size_t ocl_patch_matching::matching_policies::CLMatcher::device_id() const
{
	return impl()->device_id();
}

void ocl_patch_matching::matching_policies::CLMatcher::initialize_opencl_state(const std::shared_ptr<simple_cl::cl::Context>& clcontext)
{
	impl()->initialize_opencl_state(clcontext);
}

void ocl_patch_matching::matching_policies::CLMatcher::cleanup_opencl_state()
{
	impl()->cleanup_opencl_state();
}

void ocl_patch_matching::matching_policies::CLMatcher::compute_matches(
	const Texture& texture,
	const Texture& kernel,
	const cv::Mat& kernel_mask,
	double texture_rotation,
	MatchingResult& match_res_out)
{
	impl()->compute_matches(texture, kernel, kernel_mask, texture_rotation, match_res_out);
}

void ocl_patch_matching::matching_policies::CLMatcher::compute_matches(
	const Texture& texture,
	const Texture& kernel,
	double texture_rotation,
	MatchingResult& match_res_out)
{
	impl()->compute_matches(texture, kernel, texture_rotation, match_res_out);
}

void ocl_patch_matching::matching_policies::CLMatcher::compute_matches(
	const Texture& texture,
	const cv::Mat& texture_mask,
	const Texture& kernel,
	double texture_rotation,
	MatchingResult& match_res_out,
	bool erode_texture_mask)
{
	impl()->compute_matches(texture, texture_mask, kernel, texture_rotation, match_res_out, erode_texture_mask);
}

void ocl_patch_matching::matching_policies::CLMatcher::compute_matches(
	const Texture& texture,
	const cv::Mat& texture_mask,
	const Texture& kernel,
	const cv::Mat& kernel_mask,
	double texture_rotation,
	MatchingResult& match_res_out,
	bool erode_texture_mask)
{
	impl()->compute_matches(texture, texture_mask, kernel, kernel_mask, texture_rotation, match_res_out, erode_texture_mask);
}

cv::Vec3i ocl_patch_matching::matching_policies::CLMatcher::response_dimensions(
	const Texture& texture,
	const Texture& kernel,
	double texture_rotation) const
{
	return impl()->response_dimensions(texture, kernel, texture_rotation);
}

ocl_patch_matching::match_response_cv_mat_t ocl_patch_matching::matching_policies::CLMatcher::response_image_data_type(
	const Texture& texture,
	const Texture& kernel,
	double texture_rotation) const
{
	return impl()->response_image_data_type(texture, kernel, texture_rotation);
}
#pragma endregion