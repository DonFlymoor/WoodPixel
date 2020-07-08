#include <iostream>
#include <vector>
#include <ocl_template_matcher.hpp>
#include <matching_policies.hpp>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <chrono>

int main()
{
	std::cout << "hmmm....";
	
	ocl_template_matching::Matcher matcher(std::unique_ptr<ocl_template_matching::matching_policies::CLMatcher>(
			new ocl_template_matching::matching_policies::CLMatcher(ocl_template_matching::matching_policies::CLMatcher::DeviceSelectionPolicy::MostComputeUnits, 2000000000)
		));
	std::cout << "Weow!";
}