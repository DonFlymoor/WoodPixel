/** \file ocl_patch_matcher.hpp
*	\author Fabian Friederichs
*
*	\brief Provides an interface for patch matching implementations.
*/

#ifndef _OCL_PATCH_MATCHER_H_
#define _OCL_PATCH_MATCHER_H_

#include <memory>
#include <vector>
#include <texture.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>

/// forward declarations for interfaces
namespace simple_cl{namespace cl{class Context;}}

/**
 *  \namespace ocl_patch_matching
 *  \brief Patch matching functionality utilizing OpenCL 1.2 capabilities of the GPU.
*/
namespace ocl_patch_matching
{
    /**
     *  \struct Match
     *  \brief  Packs information about a match.
    */
    struct Match
    {
        cv::Point match_pos;        ///<    Image position of this match.
        std::size_t rotation_index; ///<    Index of the image rotation this match was found for.
        double match_cost;          ///<    Total cost of this match.
    };

    /**
     *  \struct MatchingResult
     *  \brief  Result of a single matching pass.
     *
     *  A MatchingPolicyBase implementation returns the cost matrix and a number of matches, sorted from best to worst.
    */
    struct MatchingResult
    {
        cv::Mat total_cost_matrix;      ///< Cost matrix. Has dimensions MatchingPolicyBase::response_dimensions(...).
        std::vector<Match> matches;     ///< A vector of matches, sorted from best to worst (i.e. by match_cost in ascending order)
    };

    /// OpenCV image data type alias.
    using match_response_cv_mat_t = int;

    /**
     *  \brief  Abstract base class for matching policies.
     *
     *  A matching policy implements the actual matching algorithm. When the overridden uses_opencl() returns true, the Matcher class passes a valid simple_cl::cl::Context
     *  to the initialize_opencl_state callback, which allows the implementation to use OpenCL functionality.
    */
    class MatchingPolicyBase
    {
    public:
        /// Default constructor
        MatchingPolicyBase() = default;
        MatchingPolicyBase(const MatchingPolicyBase&) = delete;
        MatchingPolicyBase(MatchingPolicyBase&&) = delete;
        MatchingPolicyBase& operator=(const MatchingPolicyBase&) = delete;
        MatchingPolicyBase& operator=(MatchingPolicyBase&&) = delete;

        inline virtual ~MatchingPolicyBase() noexcept = 0;

        /**
         *  \brief  Override and return true if the implementation requires an OpenCL context.
         *  \return Returns false if not overridden.
        */
        virtual bool uses_opencl() const { return false; }
        /**
         *  \brief  Override this function to receive an OpenCL context.
         *  \param  Shared pointer to a valid OpenCL context.
        */
        virtual void initialize_opencl_state(const std::shared_ptr<simple_cl::cl::Context>&) {}
        /**
         *  \brief  Override this function if there is OpenCL state to clean up when the Matcher instance is destroyed. 
        */
        virtual void cleanup_opencl_state() {}

        /**
         *  \brief                      Returns the dimensions of the resulting cost matrix given some texture, kernel and rotation angle in radians.
         *  \param texture              Texture instance for which the result dimensions shall be calculated.
         *  \param kernel               Kernel texture instance in case the output dimension also depends on the kernel size.
         *  \param texture_rotation     Texture rotation angle to calculate the result dimensions for.
         *  \return                     Returns a cv::Vec3i. First and second component define width and height of the resulting cost matrix, the third component tells the number of channels the response will have.
        */
        virtual cv::Vec3i response_dimensions(const Texture& texture, const Texture& kernel, double texture_rotation) const = 0;
       
        /**
         *  \brief                      Returns the OpenCV datatype used in the resulting cost matrix (e.g. CV_32FC1).
         *  \param texture              Input texture.
         *  \param kernel               Kernel texture instance.
         *  \param texture_rotation     Texture rotation.
         *  \return                     OpenCV datatype id.
        */
        virtual match_response_cv_mat_t response_image_data_type(const Texture& texture, const Texture& kernel, double texture_rotation) const = 0;

        /**
         *  \brief                      Performs one matching pass given texture, kernel and a number of rotations.
         *  \param texture              Input texture.
         *  \param kernel               Kernel or template to be searched for in texture.
         *  \param texture_rotations    Input texture rotations to try.
         *  \param[out] match_res_out   Result of the matching pass.     
        */
        virtual void compute_matches(
            const Texture& texture,
            const Texture& kernel,
            const std::vector<double>& texture_rotations,
            MatchingResult& match_res_out
        )
        {};

        /**
         *  \brief                      Performs one matching pass given texture, kernel and a number of rotations. Possible matches are masked using texture_mask.
         *  texture_mask can be any grayscale image. Every pixel in the input texture with the corresponsing mask pixel > 0 is considered as a potential match candidate.
         *  As an optional step, the mask can be eroded with the kernel bounding box as structuring element, first.
         *  \param texture              Input texture.
         *  \param texture_mask         Input texture mask. Must be a grayscale (single channel!) image of the same dimensions as texture.
         *  \param kernel               Kernel or template to be searched for in texture.
         *  \param texture_rotations    Input texture rotations to try.
         *  \param[out] match_res_out   Result of the matching pass
         *  \param erode_texture_mask   If true, the texture mask is eroded with the kernel's bounding box as structuring element before use.
        */
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

         /**
         *  \brief                      Performs one matching pass given texture, kernel and a number of rotations. The kernel is masked using kernel_mask.
         *  kernel_mask can be any grayscale image. Only kernel pixels whose corresponding kernel mask pixel is > 0 are considered for the calculation of matching costs.
         *  As an optional step, the mask can be eroded with the kernel bounding box as structuring element, first.
         *  \param texture              Input texture.
         *  \param kernel               Kernel or template to be searched for in texture.
         *  \param kernel_mask          Kernel mask. Must be a grayscale (single channel!) image of the same dimension as kernel.
         *  \param texture_rotations    Input texture rotations to try.
         *  \param[out] match_res_out   Result of the matching pass
        */
        virtual void compute_matches(
            const Texture& texture,
            const Texture& kernel,
            const cv::Mat& kernel_mask,
            const std::vector<double>& texture_rotations,
            MatchingResult& match_res_out
        )
        {
        };

         /**
         *  \brief                      Performs one matching pass given texture, kernel and a number of rotations. Possible matches are masked using texture_mask and the kernel is masked using kernel_mask.
         *  texture_mask can be any grayscale image. Every pixel in the input texture with the corresponsing mask pixel > 0 is considered as a potential match candidate.
         *  kernel_mask can be any grayscale image. Only kernel pixels whose corresponding kernel mask pixel is > 0 are considered for the calculation of matching costs.
         *  As an optional step, the mask can be eroded with the kernel bounding box as structuring element, first.
         *  \param texture              Input texture.
         *  \param texture_mask         Input texture mask. Must be a grayscale (single channel!) image of the same dimensions as texture.
         *  \param kernel               Kernel or template to be searched for in texture.
         *  \param kernel_mask          Kernel mask. Must be a grayscale (single channel!) image of the same dimension as kernel.
         *  \param texture_rotations    Input texture rotations to try.
         *  \param[out] match_res_out   Result of the matching pass
         *  \param erode_texture_mask   If true, the texture mask is eroded with the kernel mask as structuring element before use.
        */
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
    /**
     *  \brief Provides a unified interface for different matching strategies and internally manages an OpenCL context which can be used by the matching policies to utilize GPU resources.
    */
    class Matcher
    {
    public:
        /**
         *  \brief  Specifies how the GPU will be selected if there are more than one. 
        */
        enum class DeviceSelectionPolicy
        {
            MostComputeUnits,       ///< The GPU with the most compute units will be selected.
            MostGPUThreads,         ///< The GPU with the most available threads will be selected.
            FirstSuitableDevice     ///< The first available GPU with OpenCL 1.2 support will be selected.
        };

        /**
         *  \brief                          Creates a new matcher instance which uses matching_policy to do the actual matching.
         *  \param matching_policy          Implementation of the matching algorithm.
         *  \param device_selection_policy  Defines how a GPU in the system is selected.
        */
        Matcher(std::unique_ptr<MatchingPolicyBase>&& matching_policy, DeviceSelectionPolicy device_selection_policy);
    
    public:
        /// No copies
        Matcher(const Matcher&) = delete;
        /// Move constructor
        Matcher(Matcher&& other) noexcept;
        /// No copies
        Matcher& operator=(const Matcher&) = delete;
        /// Move assignment
        Matcher& operator=(Matcher&& other) noexcept;
        /// Destructor
        ~Matcher() noexcept;

        /**
         *  \brief                      Performs one matching pass given texture, kernel and texture rotation.
         *  \param texture              Input texture.
         *  \param kernel               Kernel or template to be searched for in texture.
         *  \param texture_rotation     Input texture rotation.
         *  \param[out] match_res_out   Result of the matching pass.
        */
        void match(const Texture& texture, const Texture& kernel, double texture_rotation, MatchingResult& result);
        
        /**
         *  \brief                      Performs one matching pass given texture, kernel and a number of rotations.
         *  \param texture              Input texture.
         *  \param kernel               Kernel or template to be searched for in texture.
         *  \param texture_rotations    Input texture rotations to try.
         *  \param[out] match_res_out   Result of the matching pass.
        */
        void match(const Texture& texture, const Texture& kernel, const std::vector<double>& texture_rotations, MatchingResult& result);
       
       /**
         *  \brief                      Performs one matching pass given texture, kernel and texture rotation. Possible matches are masked using texture_mask.
         *  texture_mask can be any grayscale image. Every pixel in the input texture with the corresponsing mask pixel > 0 is considered as a potential match candidate.
         *  As an optional step, the mask can be eroded with the kernel bounding box as structuring element, first.
         *  \param texture              Input texture.
         *  \param texture_mask         Input texture mask. Must be a grayscale (single channel!) image of the same dimensions as texture.
         *  \param kernel               Kernel or template to be searched for in texture.
         *  \param texture_rotation    Input texture rotation.
         *  \param[out] match_res_out   Result of the matching pass
         *  \param erode_texture_mask   If true, the texture mask is eroded with the kernel's bounding box as structuring element before use.
        */
        void match(const Texture& texture, const cv::Mat& texture_mask, const Texture& kernel, double texture_rotation, MatchingResult& result, bool erode_texture_mask = true);
        
         /**
         *  \brief                      Performs one matching pass given texture, kernel and a number of rotations. Possible matches are masked using texture_mask.
         *  texture_mask can be any grayscale image. Every pixel in the input texture with the corresponsing mask pixel > 0 is considered as a potential match candidate.
         *  As an optional step, the mask can be eroded with the kernel bounding box as structuring element, first.
         *  \param texture              Input texture.
         *  \param texture_mask         Input texture mask. Must be a grayscale (single channel!) image of the same dimensions as texture.
         *  \param kernel               Kernel or template to be searched for in texture.
         *  \param texture_rotations    Input texture rotations to try.
         *  \param[out] match_res_out   Result of the matching pass
         *  \param erode_texture_mask   If true, the texture mask is eroded with the kernel's bounding box as structuring element before use.
        */
        void match(const Texture& texture, const cv::Mat& texture_mask, const Texture& kernel, const std::vector<double>& texture_rotations, MatchingResult& result, bool erode_texture_mask = true);
        
        /**
         *  \brief                      Performs one matching pass given texture, kernel and texture rotations. Possible matches are masked using texture_mask.
         *  texture_mask can be any grayscale image. Every pixel in the input texture with the corresponsing mask pixel > 0 is considered as a potential match candidate.
         *  As an optional step, the mask can be eroded with the kernel bounding box as structuring element, first.
         *  \param texture              Input texture.
         *  \param texture_mask         Input texture mask. Must be a grayscale (single channel!) image of the same dimensions as texture.
         *  \param kernel               Kernel or template to be searched for in texture.
         *  \param texture_rotation     Input texture rotation.
         *  \param[out] match_res_out   Result of the matching pass
         *  \param erode_texture_mask   If true, the texture mask is eroded with the kernel's bounding box as structuring element before use.
        */
        void match(const Texture& texture, const Texture& kernel, const cv::Mat& kernel_mask, double texture_rotation, MatchingResult& result);

       /**
         *  \brief                      Performs one matching pass given texture, kernel and a number of rotations. The kernel is masked using kernel_mask.
         *  kernel_mask can be any grayscale image. Only kernel pixels whose corresponding kernel mask pixel is > 0 are considered for the calculation of matching costs.
         *  As an optional step, the mask can be eroded with the kernel bounding box as structuring element, first.
         *  \param texture              Input texture.
         *  \param kernel               Kernel or template to be searched for in texture.
         *  \param kernel_mask          Kernel mask. Must be a grayscale (single channel!) image of the same dimension as kernel.
         *  \param texture_rotations    Input texture rotations to try.
         *  \param[out] match_res_out   Result of the matching pass
        */
        void match(const Texture& texture, const Texture& kernel, const cv::Mat& kernel_mask, const std::vector<double>& texture_rotations, MatchingResult& result);

        /**
         *  \brief                      Performs one matching pass given texture, kernel and texture rotation. Possible matches are masked using texture_mask and the kernel is masked using kernel_mask.
         *  texture_mask can be any grayscale image. Every pixel in the input texture with the corresponsing mask pixel > 0 is considered as a potential match candidate.
         *  kernel_mask can be any grayscale image. Only kernel pixels whose corresponding kernel mask pixel is > 0 are considered for the calculation of matching costs.
         *  As an optional step, the mask can be eroded with the kernel bounding box as structuring element, first.
         *  \param texture              Input texture.
         *  \param texture_mask         Input texture mask. Must be a grayscale (single channel!) image of the same dimensions as texture.
         *  \param kernel               Kernel or template to be searched for in texture.
         *  \param kernel_mask          Kernel mask. Must be a grayscale (single channel!) image of the same dimension as kernel.
         *  \param texture_rotation     Input texture rotation.
         *  \param[out] match_res_out   Result of the matching pass
         *  \param erode_texture_mask   If true, the texture mask is eroded with the kernel mask as structuring element before use.
        */
        void match(const Texture& texture, const cv::Mat& texture_mask, const Texture& kernel, const cv::Mat& kernel_mask, double texture_rotation, MatchingResult& result, bool erode_texture_mask = true);

        /**
         *  \brief                      Performs one matching pass given texture, kernel and a number of rotations. Possible matches are masked using texture_mask and the kernel is masked using kernel_mask.
         *  texture_mask can be any grayscale image. Every pixel in the input texture with the corresponsing mask pixel > 0 is considered as a potential match candidate.
         *  kernel_mask can be any grayscale image. Only kernel pixels whose corresponding kernel mask pixel is > 0 are considered for the calculation of matching costs.
         *  As an optional step, the mask can be eroded with the kernel bounding box as structuring element, first.
         *  \param texture              Input texture.
         *  \param texture_mask         Input texture mask. Must be a grayscale (single channel!) image of the same dimensions as texture.
         *  \param kernel               Kernel or template to be searched for in texture.
         *  \param kernel_mask          Kernel mask. Must be a grayscale (single channel!) image of the same dimension as kernel.
         *  \param texture_rotations    Input texture rotations to try.
         *  \param[out] match_res_out   Result of the matching pass
         *  \param erode_texture_mask   If true, the texture mask is eroded with the kernel mask as structuring element before use.
        */
        void match(const Texture& texture, const cv::Mat& texture_mask, const Texture& kernel, const cv::Mat& kernel_mask, const std::vector<double>& texture_rotations, MatchingResult& result, bool erode_texture_mask = true);
        
        /**
         *  \brief Returns a reference to the concrete matching policy instance.
         *  \tparam ConcretePolicy Concrete policy type.
         *  \return Reference to conrete policy instance.
        */
        template <typename ConcretePolicy>
        ConcretePolicy& get_policy()
        {
            return &dynamic_cast<ConcretePolicy*>(m_matching_policy.get());
        }

        /**
         *  \brief Returns a const reference to the concrete matching policy instance.
         *  \tparam ConcretePolicy Concrete policy type.
         *  \return Reference to conrete policy instance.
        */
        template <typename ConcretePolicy>
        const ConcretePolicy& get_policy() const
        {
            return &dynamic_cast<const ConcretePolicy*>(m_matching_policy.get());
        }

    private:
        /// for const correctness
        const impl::MatcherImpl* impl() const { return m_impl.get(); }
        /// for const correctness
        impl::MatcherImpl* impl() { return m_impl.get(); }        
       
        /// pointer to implementation
        std::unique_ptr<impl::MatcherImpl> m_impl;
        /// matching policy
        std::unique_ptr<MatchingPolicyBase> m_matching_policy;
    };       
}
#endif