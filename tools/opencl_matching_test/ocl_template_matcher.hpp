#ifndef _OCL_TEMPLATE_MATCHER_H_
#define _OCL_TEMPLATE_MATCHER_H_

#include <memory>

namespace ocl_template_matching
{
    namespace impl { class MatcherImpl; }

    /**
    * \brief Implements flexible image template matching using OpenCL.
    *
    * This class implements template matching for images using OpenCL 1.2 capabilities of available graphics hardware.
    *
    * ================ UNDER CONSTRUCTION =========================================================
    * What do we need?
    *       input images can change size and format. Provide function which generates some kind of handle to the
    *       gpu ressources generated for some input texture. If this handle is passed to the main match function
    *       the buffers etc. don't have to be recreated. Maybe management (paging?) of gpu mem is necessary.

    *       PARAMETERS:
    *           Constant for Matcher instance:
    *               +   matching policy
    *                   +   work group size
    *                   +   opencl kernel for calculating response
    *                   +   host code
    *                   +   maximum extraction policy
    *                       +   work group size
    *                       +   kernel
    *                       +   host code
    *           Variable per match call:
    *               +   Input textures + format
    *               +   Mask
    *               +   Input patch + format
    *               +   Matching policy specific params
    *               +   Output (defined by input? => N-layer texture -> N-layer input patch)
    */     
    
    template <class MatchingPolicy>
    class Matcher
    {
    public:
        
    private:
        std::unique_ptr <ocl_template_matching::impl::MatcherImpl> m_impl;
        MatchingPolicy m_matchstrat;
    };
}
#endif