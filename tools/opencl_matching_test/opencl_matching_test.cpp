#include <iostream>
#include <vector>
#include <ocl_template_matcher.hpp>
#include <matching_policies.hpp>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <chrono>
#include <feature_evaluator.hpp>
#include <gabor_filter.hpp>
#include <gabor_filter_bank.hpp>

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
	
	ocl_template_matching::Matcher matcher(std::unique_ptr<ocl_template_matching::matching_policies::CLMatcher>(
			new ocl_template_matching::matching_policies::CLMatcher(ocl_template_matching::matching_policies::CLMatcher::DeviceSelectionPolicy::MostComputeUnits, 2000000000, 16ull, 256ull * 256ull)
		));

	double scale{0.25};
	Texture input_tex("img/lcds.jpg", 96.0, scale);
	Texture kernel_tex("img/lcds_res_kernel.jpg", 96.0, scale);
	cv::Mat kernel_mask_bgr{cv::imread("img/lcds_res_kernel_mask.jpg")};
	cv::Mat kernel_mask_small_bgr;
	cv::resize(kernel_mask_bgr, kernel_mask_small_bgr, cv::Size{}, scale, scale);
	cv::Mat kernel_mask;
	cv::cvtColor(kernel_mask_small_bgr, kernel_mask, CV_BGR2GRAY);
	// apply feature filters
	GaborFilterBank gfbank(32, 1.0, 4);
	FeatureEvaluator feval(0.5, 0.5, 0.0, gfbank);
	cv::Mat kernel = cv::Mat::ones(feval.max_filter_size(), CV_8UC1);

	input_tex.response = feval.evaluate(input_tex.texture, input_tex.mask());
	cv::erode(input_tex.mask_rotation, input_tex.mask_rotation, kernel, cv::Point(-1, -1), 1, cv::BORDER_CONSTANT, cv::Scalar(0));

	kernel_tex.response = feval.evaluate(kernel_tex.texture, kernel_tex.mask());
	cv::erode(kernel_tex.mask_rotation, kernel_tex.mask_rotation, kernel, cv::Point(-1, -1), 1, cv::BORDER_CONSTANT, cv::Scalar(0));

	std::cout << "Kernel size: " << kernel_tex.response.cols() << " x " << kernel_tex.response.rows() << std::endl;
	std::cout << "Texture size: " << input_tex.response.cols() << " x " << input_tex.response.rows() << std::endl;

	display_image("image_orig", input_tex.texture);
	/*display_image("image_intensity", input_tex.response[0]);
	display_image("image_sobel", input_tex.response[1]);
	display_image("image_mask_rotation", input_tex.mask_rotation);
	display_image("image_mask_done", input_tex.mask_done, true);*/

	display_image("kernel_orig", kernel_tex.texture);
	/*display_image("kernel_intensity", kernel_tex.response[0]);
	display_image("kernel_sobel", kernel_tex.response[1]);
	display_image("kernel_mask_rotation", kernel_tex.mask_rotation);
	display_image("kernel_mask_done", kernel_tex.mask_done, true);*/
	int num_iters{50};

	std::cout << "OpenCV without mask...\n";
	cv::Mat rescv;
	for(int i = 0; i < num_iters; ++i)
	{
		auto t1{std::chrono::high_resolution_clock::now()};
		rescv = input_tex.template_match(kernel_tex);
		auto mscv{std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - t1).count()};
		std::cout << mscv << " us.\n";
	}
	cv::Mat rescv_f(rescv.rows, rescv.cols, CV_32FC1);
	rescv.convertTo(rescv_f, CV_32FC1);	

	display_intensity("ResultCV", rescv_f, false);

	std::cout << "OpenCL without mask...\n";
	ocl_template_matching::MatchingResult result;
	for(int i = 0; i < num_iters; ++i)
	{
		auto t1 = std::chrono::high_resolution_clock::now();
		matcher.match(input_tex, kernel_tex, 0.0, result);
		auto mscl{std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - t1).count()};
		std::cout << mscl << " us.\n";
	}
	display_intensity("ResultCL", result.total_cost_matrix, true);
	cv::destroyAllWindows();

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
		std::cout << mscl << " us.\n";
	}
	display_intensity("ResultCL", result.total_cost_matrix, true);
	
	
}