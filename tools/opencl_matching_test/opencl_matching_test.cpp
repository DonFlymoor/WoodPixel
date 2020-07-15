#include <iostream>
#include <vector>
#include <ocl_patch_matcher.hpp>
#include <matching_policies.hpp>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <chrono>
#include <feature_evaluator.hpp>
#include <gabor_filter.hpp>
#include <gabor_filter_bank.hpp>

#define PIf 3.14159265359f

void display_image(const std::string& name, const cv::Mat& mat, bool wait = false)
{
	cv::imshow(name, mat);
	if(wait)
		cv::waitKey();
}

void display_intensity(const std::string& name, const cv::Mat& mat, bool wait = false)
{
	double maxval;
	double minval;
	cv::minMaxLoc(mat, &minval, &maxval);
	cv::Mat newmat((mat - minval) / (maxval - minval));
	cv::imshow(name, newmat);
	if(wait)
		cv::waitKey();
}

int main()
{
	std::cout << "hmmm....";
	
	ocl_patch_matching::Matcher matcher(
		std::unique_ptr<ocl_patch_matching::matching_policies::CLMatcher>(
			new ocl_patch_matching::matching_policies::CLMatcher(
				ocl_patch_matching::matching_policies::CLMatcher::DeviceSelectionPolicy::MostComputeUnits,
				2000000000,
				16ull, 256ull * 256ull
			)
		));

	cv::Mat texture_mask{cv::imread("img/testmask.jpg", CV_LOAD_IMAGE_GRAYSCALE)};
	cv::Mat texture_mask_kernel{cv::imread("img/testmask_kernel.jpg", CV_LOAD_IMAGE_GRAYSCALE)};
	cv::Mat outmask;
	auto t1 = std::chrono::high_resolution_clock::now();
	matcher.erode_texture_mask(texture_mask, outmask, texture_mask_kernel, cv::Point(0, 0), 0.0);
	std::cout << "CL Version took " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t1).count() << std::endl;
	cv::Mat outmask2;
	t1 = std::chrono::high_resolution_clock::now();
	cv::erode(texture_mask, outmask2, texture_mask_kernel, cv::Point{0, 0}, 1, cv::BORDER_CONSTANT, cv::Scalar(0));
	std::cout << "CV Version took " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t1).count() << std::endl;
	display_image("Mask before", texture_mask);
	display_image("Kernel", texture_mask_kernel);
	display_intensity("Mask after, CL", outmask, false);
	display_image("Mask after, CV", outmask2, true);

	//double scale{0.16666};
	//Texture input_tex("img/lcds.jpg", 96.0, scale);
	//Texture kernel_tex("img/lcds_seg_big_kernel.jpg", 96.0, scale);
	//cv::Mat kernel_mask_bgr{cv::imread("img/lcds_seg_kernel_mask.jpg")};
	//cv::Mat kernel_mask_small_bgr;
	//cv::resize(kernel_mask_bgr, kernel_mask_small_bgr, cv::Size{}, scale, scale);
	//cv::Mat kernel_mask;
	//cv::cvtColor(kernel_mask_small_bgr, kernel_mask, CV_BGR2GRAY);
	//// apply feature filters
	//GaborFilterBank gfbank(32, 1.0, 4);
	//FeatureEvaluator feval(0.5, 0.5, 0.0, gfbank);
	//cv::Mat kernel = cv::Mat::ones(feval.max_filter_size(), CV_8UC1);

	//input_tex.response = feval.evaluate(input_tex.texture, input_tex.mask());
	//cv::erode(input_tex.mask_rotation, input_tex.mask_rotation, kernel, cv::Point(-1, -1), 1, cv::BORDER_CONSTANT, cv::Scalar(0));

	//kernel_tex.response = feval.evaluate(kernel_tex.texture, kernel_tex.mask());
	//cv::erode(kernel_tex.mask_rotation, kernel_tex.mask_rotation, kernel, cv::Point(-1, -1), 1, cv::BORDER_CONSTANT, cv::Scalar(0));

	//std::cout << "Kernel size: " << kernel_tex.response.cols() << " x " << kernel_tex.response.rows() << std::endl;
	//std::cout << "Texture size: " << input_tex.response.cols() << " x " << input_tex.response.rows() << std::endl;

	//display_image("image_orig", input_tex.texture);

	//display_image("kernel_orig", kernel_tex.texture);
	//int num_iters{50};

	//std::cout << "OpenCV without mask...\n";
	//cv::Mat rescv;
	//input_tex.rotate(-PIf / 2.0);
	//for(int i = 0; i < num_iters; ++i)
	//{
	//	auto t1{std::chrono::high_resolution_clock::now()};
	//	rescv = input_tex.template_match(kernel_tex);
	//	auto mscv{std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - t1).count()};
	//	std::cout << mscv << " us.\n";
	//}
	//cv::Mat rescv_f(rescv.rows, rescv.cols, CV_32FC1);
	//rescv.convertTo(rescv_f, CV_32FC1);	

	//display_intensity("ResultCV", rescv_f, false);

	//std::cout << "OpenCL without mask...\n";
	//ocl_patch_matching::MatchingResult result;
	//for(int i = 0; i < num_iters; ++i)
	//{
	//	auto t1 = std::chrono::high_resolution_clock::now();
	//	matcher.match(input_tex, kernel_tex, PIf, result);
	//	auto mscl{std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - t1).count()};
	//	std::cout << mscl << " us.\n";
	//}
	//display_intensity("ResultCL", result.total_cost_matrix, true);
	/*cv::destroyAllWindows();

	std::cout << "OpenCV with mask...\n";
	for(int i = 0; i < num_iters; ++i)
	{
		auto t1{std::chrono::high_resolution_clock::now()};
		rescv = input_tex.template_match(kernel_tex, kernel_mask);
		auto mscv{std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - t1).count()};
		std::cout << mscv << " us.\n";
	}
	rescv.convertTo(rescv_f, CV_32FC1);
	display_intensity("ResultCV", rescv_f, false);

	std::cout << "OpenCL with mask...\n";
	for(int i = 0; i < num_iters; ++i)
	{
		auto t1 = std::chrono::high_resolution_clock::now();
		matcher.match(input_tex, kernel_tex, kernel_mask, 0.0, result);
		auto mscl{std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - t1).count()};
		std::cout << mscl << " us template matching.\n";
		double minval, maxval;
		cv::Point minloc, maxloc;
		t1 = std::chrono::high_resolution_clock::now();
		cv::minMaxLoc(result.total_cost_matrix, &minval, &maxval, &minloc, &maxloc);
		auto mscl2{std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - t1).count()};
		std::cout << mscl2 << " us min search.\n";
	}
	display_intensity("ResultCL", result.total_cost_matrix, true);*/
	
	
}