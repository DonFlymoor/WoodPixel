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
#include <fstream>

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
				2000000000,
				16ull, 500ull * 500ull , 4096ull, 16ull,
				ocl_patch_matching::matching_policies::CLMatcher::ResultOrigin::UpperLeftCorner,
				true,
				true
			)
		), ocl_patch_matching::Matcher::DeviceSelectionPolicy::MostComputeUnits);

	double scale{0.16666};
	double rotation{0.0};
	Texture input_tex("img/furnier.jpg", 96.0, scale);
	Texture cv_input_tex{input_tex.rotate(rotation)};
	cv::Mat texture_mask_big{cv::imread("img/furnier_texture_mask.png", CV_LOAD_IMAGE_GRAYSCALE)};
	cv::Mat texture_mask;
	cv::resize(texture_mask_big, texture_mask, cv::Size{}, scale, scale);
	Texture kernel_tex("img/furnier_kernel.jpg", 96.0, scale);
	cv::Mat kernel_mask_big{cv::imread("img/furnier_kernel_mask.jpg", CV_LOAD_IMAGE_GRAYSCALE)};
	cv::Mat kernel_mask;
	cv::resize(kernel_mask_big, kernel_mask, cv::Size{}, scale, scale);
	
	// apply feature filters
	GaborFilterBank gfbank(32, 1.0, 4);
	FeatureEvaluator feval(0.5, 0.5, 0.0, gfbank);
	cv::Mat kernel = cv::Mat::ones(feval.max_filter_size(), CV_8UC1);

	input_tex.response = feval.evaluate(input_tex.texture, input_tex.mask());
	cv::erode(input_tex.mask_rotation, input_tex.mask_rotation, kernel, cv::Point(-1, -1), 1, cv::BORDER_CONSTANT, cv::Scalar(0));

	cv_input_tex.response = feval.evaluate(cv_input_tex.texture, cv_input_tex.mask());
	cv::erode(cv_input_tex.mask_rotation, cv_input_tex.mask_rotation, kernel, cv::Point(-1, -1), 1, cv::BORDER_CONSTANT, cv::Scalar(0));

	kernel_tex.response = feval.evaluate(kernel_tex.texture, kernel_tex.mask());
	cv::erode(kernel_tex.mask_rotation, kernel_tex.mask_rotation, kernel, cv::Point(-1, -1), 1, cv::BORDER_CONSTANT, cv::Scalar(0));

	cv::Mat eroded_mask_cv;
	cv::erode(texture_mask, eroded_mask_cv, kernel_mask, cv::Point(0, 0), 1, cv::BORDER_CONSTANT, cv::Scalar(0));

	std::cout << "Texture size: " << input_tex.response.cols() << " x " << input_tex.response.rows() << std::endl;

	std::size_t max_kernel_size = 128;

	cv::Point kernel_pos = cv::Point((input_tex.response.cols() - 1) / 2, (input_tex.response.cols() - 1) / 2);

	cv::Mat rescv;
	cv::Point minpos;
	double minval;
	std::cout << "OpenCV with kernel mask and texture mask...\n";
	std::ofstream pdataout_cv("perf_data_cv.csv", std::ios_base::binary | std::ios_base::out | std::ios_base::trunc);
	for(int i = 2; i <= max_kernel_size; ++i)
	{
		Texture kernel_tex_sc = input_tex(cv::Rect(kernel_pos.x - i / 2, kernel_pos.y - i / 2, i, i));
		cv::Mat kernel_mask_sc;
		cv::resize(kernel_mask, kernel_mask_sc, cv::Size(kernel_tex_sc.response.cols(), kernel_tex_sc.response.rows()));		
		auto t1{std::chrono::high_resolution_clock::now()};
		cv::Mat texture_mask_cv{eroded_mask_cv(cv::Rect(0, 0, eroded_mask_cv.cols - kernel_tex_sc.response.cols() + 1, eroded_mask_cv.rows - kernel_tex_sc.response.rows() + 1))};
		rescv = cv_input_tex.template_match(kernel_tex_sc, kernel_mask_sc);
		cv::minMaxLoc(rescv, &minval, nullptr, &minpos, nullptr, texture_mask_cv);
		auto mscv{std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - t1).count()};
		std::cout << "Kernel size: " << kernel_tex_sc.response.cols() << " x " << kernel_tex_sc.response.rows() << std::endl;
		std::cout << mscv << " us. Min pos: " << "x " << minpos.x << " y " << minpos.y << " cost " << minval << "\n";
		pdataout_cv << i << ", " << mscv << "\n";
	}
	

	std::cout << "OpenCL with kernel mask and texture mask...\n";
	ocl_patch_matching::MatchingResult result;
	std::ofstream pdataout_cl("perf_data_cl.csv", std::ios_base::binary | std::ios_base::out | std::ios_base::trunc);
	for(int i = 2; i <= max_kernel_size; ++i)
	{
		Texture kernel_tex_sc = input_tex(cv::Rect(kernel_pos.x - i / 2, kernel_pos.y - i / 2, i, i));
		cv::Mat kernel_mask_sc;
		cv::resize(kernel_mask, kernel_mask_sc, cv::Size(kernel_tex_sc.response.cols(), kernel_tex_sc.response.rows()));
		auto t1 = std::chrono::high_resolution_clock::now();
		matcher.match(input_tex, texture_mask, kernel_tex_sc, kernel_mask_sc, rotation, result, true);
		auto mscl{std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - t1).count()};
		std::cout << "Kernel size: " << kernel_tex_sc.response.cols() << " x " << kernel_tex_sc.response.rows() << std::endl;
		std::cout << mscl << " us. Min pos: " << "x " << result.matches[0].match_pos.x << " y " << result.matches[0].match_pos.y << " cost " << result.matches[0].match_cost << "\n";
		pdataout_cl << i << ", " << mscl << "\n";
	}	
}