#ifndef _OCL_TEMPLATE_MATCHER_H_
#define _OCL_TEMPLATE_MATCHER_H_

//TODO: design this class:
// OpenCL Platform, Context, Device, Buffers, Images, Kernels
// Input Image format, input image buffer
// Filter response buffer
// Maxima Detection
// OpenCL: Number of Workgroups, Workgroup size...
// Matching stragies via Policy
//		cross correlation
//		normalized cross correlation
//		correlation coefficient
//		normalized correlation coefficient



#include <memory>

namespace ocl_template_matching
{
    namespace impl { class MatcherImpl; }

    /**
    * \brief Implements flexible image template matching using OpenCL.
    *
    * This class implements template matching for images using OpenCL 1.2 capabilities of available graphics hardware.
    *
    */     
    template <typename MatchingStrategy>
    class Matcher : public MatchingStrategy
    {
    public:

    private:
        std::unique_ptr <ocl_template_matching::impl::MatcherImpl> m_impl;
    };
}
#endif