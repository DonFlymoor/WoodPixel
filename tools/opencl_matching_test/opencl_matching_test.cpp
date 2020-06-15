#include <iostream>
#include <vector>
#include <cassert>
#include <ocl_template_matcher.hpp>
#include <matching_policies.hpp>


int main()
{
	try
	{
		ocl_template_matching::Matcher<DefaultMatchingPolicy> matcher(DefaultMatchingPolicy{0, 0});
	}
	catch(const std::exception& ex)
	{
		std::cerr << "ERROR: " << ex.what() << std::endl;
	}	
	return 0;
}