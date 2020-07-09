// sampler

__kernel void sqdiff_naive_both_masks(
	__read_only image2d_array_t input_tex,
	__read_only image2d_array_t kernel_tex,
	__read_only image2d_t input_mask,
	__read_only image2d_t kernel_mask,
	__write_only image2d_t response_tex,
	float rotation)
{
	printf("Hello CL!");		
}

__kernel void sqdiff_naive_tex_mask(
	__read_only image2d_array_t input_tex,
	__read_only image2d_array_t kernel_tex,
	__read_only image2d_t input_mask,
	__write_only image2d_t response_tex,
	float rotation)
{
	printf("Hello CL!");		
}

__kernel void sqdiff_naive_kernel_mask(
	__read_only image2d_array_t input_tex,
	__read_only image2d_array_t kernel_tex,
	__read_only image2d_t kernel_mask,
	__write_only image2d_t response_tex,
	float rotation)
{
	printf("Hello CL!");		
}

__kernel void sqdiff_naive_no_mask(
	__read_only image2d_array_t input_tex,
	__read_only image2d_array_t kernel_tex,
	__write_only image2d_t response_tex,
	float rotation)
{
	printf("Hello CL!");		
}