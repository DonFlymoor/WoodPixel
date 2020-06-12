#ifndef _OCL_TEMPLATE_MATCHER_H_
#define _OCL_TEMPLATE_MATCHER_H_

#include <memory>
#include <vector>
#include <texture.hpp>
#include <opencv2/opencv.hpp>

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

namespace ocl_template_matching
{
    struct MatchingResult
    {
        cv::Mat total_cost_matrix;
        cv::Point min_cost_pos;
        double min_cost;
    };

    class MatchingStrategyBase
    {

    };

    class MatchExtractionStrategyBase
    {

    };

    namespace impl
    { 
        class MatcherImpl;
        class MatcherBase
        {
        protected:
            MatcherBase(const ocl_template_matching::MatchingStrategyBase& matching_strat, const ocl_template_matching::MatchExtractionStrategyBase& matchex_strat);
            MatcherBase(const MatcherBase&) = delete;
            MatcherBase(MatcherBase&& other) noexcept;
            MatcherBase& operator=(const MatcherBase&) = delete;
            MatcherBase& operator=(MatcherBase&& other) noexcept;
            ~MatcherBase();

            MatchingResult match(const Texture& texture, const cv::Mat& texture_mask, const Texture& kernel, const cv::Mat& kernel_mask, const MatchingStrategyBase& matching_strat, const MatchExtractionStrategyBase& matchex_strat);
            void match(const Texture& texture, const cv::Mat& texture_mask, const Texture& kernel, const cv::Mat& kernel_mask, MatchingResult& result, const MatchingStrategyBase& matching_strat, const MatchExtractionStrategyBase& matchex_strat);

        private:
            std::unique_ptr <ocl_template_matching::impl::MatcherImpl> m_impl;
        };      

    }

    template <typename MatchingStrategy, typename MatchExtractionStrategy>
    class Matcher : private MatcherBase
    {
    public:
        Matcher(const MatchingStrategy& matching_strat, const MatchExtractionStrategy& matchex_strat);
        Matcher(const Matcher&) = delete;
        Matcher(Matcher&& other) = default;
        Matcher& operator=(const Matcher&) = delete;
        Matcher& operator=(Matcher&& other) = default;
        ~Matcher();

        MatchingResult match(const Texture& texture, const cv::Mat& texture_mask, const Texture& kernel, const cv::Mat& kernel_mask);
        void match(const Texture& texture, const cv::Mat& texture_mask, const Texture& kernel, const cv::Mat& kernel_mask, MatchingResult& result);
    
    private:
        MatchingStrategy m_matching_strategy;
        MatchExtractionStrategy m_match_extraction_strategy;
    };

    template <typename MatchingStrategy, typename MatchExtractionStrategy>
    Matcher<MatchingStrategy, MatchExtractionStrategy>::Matcher(const MatchingStrategy& matching_strat, const MatchExtractionStrategy& matchex_strat) :
        MatcherBase(matching_strat, matchex_strat),
        m_matching_strategy(matching_strat),
        m_match_extraction_strategy(matchex_strat)
    {
    }

    template <typename MatchingStrategy, typename MatchExtractionStrategy>
    Matcher<MatchingStrategy, MatchExtractionStrategy>::~Matcher()
    {
    }

    template <typename MatchingStrategy, typename MatchExtractionStrategy>
    MatchingResult Matcher<MatchingStrategy, MatchExtractionStrategy>::match(const Texture& texture, const cv::Mat& texture_mask, const Texture& kernel, const cv::Mat& kernel_mask)
    {
        return impl::MatcherBase::match(texture, texture_mask, kernel, kernel_mask, m_matching_strategy, m_match_extraction_strategy);
    }

    template <typename MatchingStrategy, typename MatchExtractionStrategy>
    void Matcher<MatchingStrategy, MatchExtractionStrategy>::match(const Texture& texture, const cv::Mat& texture_mask, const Texture& kernel, const cv::Mat& kernel_mask, MatchingResult& result)
    {
        result = impl::MatcherBase::match(texture, texture_mask, kernel, kernel_mask, m_matching_strategy, m_match_extraction_strategy);
    }        
}
#endif