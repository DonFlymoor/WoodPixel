#include <iostream>
#include <vector>
//#include <ocl_template_matcher.hpp>
//#include <matching_policies.hpp>
#include <simple_cl.hpp>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <chrono>

using namespace simple_cl;

int main()
{
	try
	{
		std::cout << "Creating OpenCL Instance...";
		auto clplatform{cl::Context::createInstance(0, 0)};
		std::cout << "done!." << std::endl;
		
		std::string kernel_src = R"(
			__constant sampler_t image_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

			kernel void hello_world(__read_only image2d_t image_in, __write_only image2d_t image_out)
			{
				size_t gidx = get_global_id(0);
				size_t gidy = get_global_id(1);
				float4 incol = read_imagef(image_in, image_sampler, (int2)(gidx, gidy));
				write_imagef(image_out, (int2)(gidx, gidy), (float4)(255.0f - incol[0], 255.0f - incol[1], 255.0f - incol[2], incol[3]));				
			}
		)";
		std::cout << "Loading image using OpenCV...";
		cv::Mat image{cv::imread("wooly.jpg", cv::ImreadModes::IMREAD_COLOR)};
		cv::imshow("Input", image);
		cv::waitKey();
		cv::Mat image_f_bgr;
		image.convertTo(image_f_bgr, CV_32FC3);
		cv::Mat image_rgba{image.rows, image.cols, CV_32FC4};
		cv::cvtColor(image_f_bgr, image_rgba, CV_BGR2RGBA);
		std::cout << "Creating device images...";
		cl::Image::ImageDesc idesc_in{
			cl::Image::ImageType::Image2D,
			cl::Image::ImageDimensions{static_cast<std::size_t>(image_rgba.cols), static_cast<std::size_t>(image_rgba.rows)},
			cl::Image::ImageChannelOrder::RGBA,
			cl::Image::ImageChannelType::FLOAT,
			cl::MemoryFlags{
				cl::DeviceAccess::ReadOnly,
				cl::HostAccess::WriteOnly,
				cl::HostPointerOption::None
			},
			cl::Image::HostPitch{},
			nullptr
		};

		cl::Image::ImageDesc idesc_out{
			cl::Image::ImageType::Image2D,
			cl::Image::ImageDimensions{static_cast<std::size_t>(image_rgba.cols), static_cast<std::size_t>(image_rgba.rows)},
			cl::Image::ImageChannelOrder::RGBA,
			cl::Image::ImageChannelType::FLOAT,
			cl::MemoryFlags{
				cl::DeviceAccess::WriteOnly,
				cl::HostAccess::ReadOnly,
				cl::HostPointerOption::None
			},
			cl::Image::HostPitch{},
			nullptr
		};

		cl::Image climg_in(clplatform, idesc_in);
		cl::Image climg_out(clplatform, idesc_out);

		std::cout << "done!" << std::endl;
		// upload image data
		std::cout << "Uploading image data to gpu...";
		auto t1 = std::chrono::high_resolution_clock::now();
		climg_in.write(
			cl::Image::ImageRegion{
				cl::Image::ImageOffset{},
				cl::Image::ImageDimensions{static_cast<std::size_t>(image_rgba.cols), static_cast<std::size_t>(image_rgba.rows)},
			},
			cl::Image::HostFormat{
				cl::Image::HostChannelOrder{4, {cl::Image::ColorChannel::R, cl::Image::ColorChannel::G, cl::Image::ColorChannel::B, cl::Image::ColorChannel::A}},
				cl::Image::HostDataType::FLOAT,
				cl::Image::HostPitch{static_cast<std::size_t>(image_rgba.step[0]), 0ull}
			},
			static_cast<const void*>(image_rgba.data),
			true
		).wait();
		auto t = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t1).count();
		std::cout << "done! Time: " << t << " ms" << std::endl;
		std::cout << "Compiling kernels...";
		auto progra{cl::Program(kernel_src, "", clplatform)};
		std::cout << "done!" << std::endl;

		std::cout << "Executing kernel...";
		cl::Program::ExecParams exec_params{
			2,
			{0, 0, 0},
			{image_rgba.cols, image_rgba.rows, 1},
			{8, 8, 0}
		};
		t1 = std::chrono::high_resolution_clock::now();
		progra("hello_world", exec_params, climg_in, climg_out).wait();
		t = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t1).count();
		std::cout << "done! Time: " << t << " ms" << std::endl;

		std::cout << "Reading result...";
		t1 = std::chrono::high_resolution_clock::now();
		climg_out.read(
			cl::Image::ImageRegion{
				cl::Image::ImageOffset{},
				cl::Image::ImageDimensions{static_cast<std::size_t>(image_rgba.cols), static_cast<std::size_t>(image_rgba.rows)},
			},
			cl::Image::HostFormat{
				cl::Image::HostChannelOrder{4, {cl::Image::ColorChannel::R, cl::Image::ColorChannel::G, cl::Image::ColorChannel::B, cl::Image::ColorChannel::A}},
				cl::Image::HostDataType::FLOAT,
				cl::Image::HostPitch{static_cast<std::size_t>(image_rgba.step[0]), 0ull}
			},
			static_cast<void*>(image_rgba.data)
		).wait();
		t = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t1).count();
		std::cout << "done! Time: " << t << " ms" << std::endl;

		cv::Mat image_res_f_bgr;
		cv::cvtColor(image_rgba, image_res_f_bgr, CV_RGBA2BGR);
		cv::Mat image_res;
		image_res_f_bgr.convertTo(image_res, CV_8UC3);
		cv::imshow("Output", image_res);
		cv::waitKey();
	}
	catch(const std::exception& ex)
	{
		std::cerr << "[ERROR]" << ex.what() << std::endl;
	}	
	return 0;
}