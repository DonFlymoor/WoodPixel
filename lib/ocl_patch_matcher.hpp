#ifndef _OCL_PATCH_MATCHER_H_
#define _OCL_PATCH_MATCHER_H_

#include <memory>
#include <vector>
#include <texture.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>

/**
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
*       For large images:
*           Pool of chunks
*               Base size + padding
*           Input, Output chunks
*       Build cache:
*           Textureid + region -> chunk id
*/ 

// forward declarations for interfaces
namespace simple_cl{namespace cl{class Context;}}

namespace ocl_patch_matching
{
    struct Match
    {
        cv::Point match_pos;
        double match_cost;
    };

    struct MatchingResult
    {
        cv::Mat total_cost_matrix;
        std::vector<Match> matches;
    };

    using match_response_cv_mat_t = int;

    class MatchingPolicyBase
    {
    public:
        MatchingPolicyBase() = default;
        MatchingPolicyBase(const MatchingPolicyBase&) = delete;
        MatchingPolicyBase(MatchingPolicyBase&&) = delete;
        MatchingPolicyBase& operator=(const MatchingPolicyBase&) = delete;
        MatchingPolicyBase& operator=(MatchingPolicyBase&&) = delete;

        inline virtual ~MatchingPolicyBase() noexcept = 0;

        // matching policy interface
        virtual std::size_t platform_id() const { return 0ull; }
        virtual std::size_t device_id() const { return 0ull; }        

        // tell if this policy uses OpenCL
        virtual bool uses_opencl() const { return false; }
        virtual void initialize_opencl_state(const std::shared_ptr<simple_cl::cl::Context>&) {}
        virtual void cleanup_opencl_state() {}

        // calculate response dimensions
        virtual cv::Vec3i response_dimensions(const Texture& texture, const Texture& kernel, double texture_rotation) const = 0;
        // report OpenCV datatype for response mat
        virtual match_response_cv_mat_t response_image_data_type(const Texture& texture, const Texture& kernel, double texture_rotation) const = 0;

        virtual void compute_matches(
            const Texture& texture,
            const Texture& kernel,
            const std::vector<double>& texture_rotations,
            MatchingResult& match_res_out
        )
        {};

        virtual void compute_matches(
            const Texture& texture,
            const cv::Mat& texture_mask,
            const Texture& kernel,
            const std::vector<double>& texture_rotations,
            MatchingResult& match_res_out,
            bool erode_texture_mask = true
        )
        {
        };

        virtual void compute_matches(
            const Texture& texture,
            const Texture& kernel,
            const cv::Mat& kernel_mask,
            const std::vector<double>& texture_rotations,
            MatchingResult& match_res_out
        )
        {
        };

        virtual void compute_matches(
            const Texture& texture,
            const cv::Mat& texture_mask,
            const Texture& kernel,
            const cv::Mat& kernel_mask,
            const std::vector<double>& texture_rotations,
            MatchingResult& match_res_out,
            bool erode_texture_mask = true
        )
        {
        };
    };
    MatchingPolicyBase::~MatchingPolicyBase() noexcept { cleanup_opencl_state(); }

    namespace impl{class MatcherImpl;}
    class Matcher
    {
    public:
        Matcher(std::unique_ptr<MatchingPolicyBase>&& matching_policy);
    
    public:
        Matcher(const Matcher&) = delete;
        Matcher(Matcher&& other) noexcept;
        Matcher& operator=(const Matcher&) = delete;
        Matcher& operator=(Matcher&& other) noexcept;
        ~Matcher() noexcept;

        void match(const Texture& texture, const Texture& kernel, double texture_rotation, MatchingResult& result);
        void match(const Texture& texture, const Texture& kernel, const std::vector<double>& texture_rotations, MatchingResult& result);
       
        void match(const Texture& texture, const cv::Mat& texture_mask, const Texture& kernel, double texture_rotation, MatchingResult& result, bool erode_texture_mask = true);
        void match(const Texture& texture, const cv::Mat& texture_mask, const Texture& kernel, const std::vector<double>& texture_rotations, MatchingResult& result, bool erode_texture_mask = true);
        
        void match(const Texture& texture, const Texture& kernel, const cv::Mat& kernel_mask, double texture_rotation, MatchingResult& result);
        void match(const Texture& texture, const Texture& kernel, const cv::Mat& kernel_mask, const std::vector<double>& texture_rotations, MatchingResult& result);

        void match(const Texture& texture, const cv::Mat& texture_mask, const Texture& kernel, const cv::Mat& kernel_mask, double texture_rotation, MatchingResult& result, bool erode_texture_mask = true);
        void match(const Texture& texture, const cv::Mat& texture_mask, const Texture& kernel, const cv::Mat& kernel_mask, const std::vector<double>& texture_rotations, MatchingResult& result, bool erode_texture_mask = true);

       /* void erode(cv::Mat& texture_mask, )*/
        
        template <typename ConcretePolicy>
        ConcretePolicy& get_policy()
        {
            return &dynamic_cast<ConcretePolicy*>(m_matching_policy.get());
        }

        template <typename ConcretePolicy>
        const ConcretePolicy& get_policy() const
        {
            return &dynamic_cast<const ConcretePolicy*>(m_matching_policy.get());
        }

    private:
        // for const correctness
        const impl::MatcherImpl* impl() const { return m_impl.get(); }
        impl::MatcherImpl* impl() { return m_impl.get(); }        
       
        // pointer to implementation
        std::unique_ptr<impl::MatcherImpl> m_impl;
         // matching policy
        std::unique_ptr<MatchingPolicyBase> m_matching_policy;
    };       
}
#endif