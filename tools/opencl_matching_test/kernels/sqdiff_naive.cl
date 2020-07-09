// sampler (maybe a constant border color would be better? don't know yet..)
const sampler_t input_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
const sampler_t mask_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

// rotatates a sample around pivot given float2(sin(theta), cos(theta)) of the desired rotation angle
float2 rotate_sample(float2 sample, float2 pivot, float2 rotation_sincos)
{
	return (float2)(
		rotation_sincos.y * sample.x + pivot.x * (1.0 - rotation_sincos.y) + rotation_sincos.x * (pivot.y - sample.y),
		rotation_sincos.y * sample.y + pivot.y * (1.0 - rotation_sincos.y) + rotation_sincos.x * (sample.x - pivot.x)
	);
}
// returns the "center" pixel index of a kernel
int2 kernel_pivot_index(int2 kernel_size)
{
	return (int2)(
		(kernel_size.x - 1) / 2,
		(kernel_size.y - 1) / 2
	);
}

__kernel void sqdiff_naive_both_masks(
	__read_only image2d_array_t input_tex,
	__read_only image2d_array_t kernel_tex,
	__read_only image2d_t input_mask,
	__read_only image2d_t kernel_mask,
	__write_only image2d_t response_tex,
	int2 input_size,
	int2 kernel_size,
	float2 rotation_sincos)
{
	int gid_x = get_global_id(0);
	int gid_y = get_global_id(1);
	int lid_x = get_local_id(0);
	int lid_y = get_local_id(1);

	// "center" index of the kernel. This is also the offset into the input image.
	int2 kernel_pivot_idx = kernel_pivot_index(kernel_size);
	int2 kernel_start_idx = -kernel_pivot_idx;
	int2 kernel_end_idx = kernel_size - kernel_pivot_idx - 1;
	float2 input_pivot = (float2)(
		(float)(kernel_pivot_idx.x + gid_x) + 0.5f,
		(float)(kernel_pivot_idx.y + gid_y) + 0.5f 
	);
	float2 kernel_pivot = (float2)(
		(float)kernel_pivot_idx.x + 0.5f,
		(float)kernel_pivot_idx.y + 0.5f
	);
	
	float sqdiff = 0.0f;
	// for each layer
	for(int layer = 0; layer < get_image_array_size(input_tex); ++layer)
	{
		// iterate over kernel area
		for(int dy = kernel_start_idx.y; dy != kernel_end_idx.y; ++dy)
		{
			for(int dx = kernel_start_idx.x; dx != kernel_end_idx.x; ++dx)
			{
				float2 cdelta = (float2)((float)dx, (float)dy);
				float4 image_coord = (float4)(
					rotate_sample(cdelta + input_pivot, input_pivot, rotation_sincos),
					(float)layer,
					0.0f
				);
				// TODO: If kernel is small, store kernel values in local memory before doing this loop!
				float4 kernel_coord = (float4)(
					kernel_pivot + cdelta,
					(float)layer,
					0.0f
				);
				float diff = read_imagef(input_tex, input_sampler, image_coord).x - read_imagef(kernel_tex, input_sampler, kernel_coord).x;
				sqdiff += diff * diff;
			}
		}
	}

	// write result
	write_imagef(response_tex, (int2)(gid_x, gid_y), (float4)(sqdiff, 0.0f, 0.0f, 0.0f));
}

__kernel void sqdiff_naive_tex_mask(
	__read_only image2d_array_t input_tex,
	__read_only image2d_array_t kernel_tex,
	__read_only image2d_t input_mask,
	__write_only image2d_t response_tex,
	int2 input_size,
	int2 kernel_size,
	float2 rotation_sincos)
{
	int gid_x = get_global_id(0);
	int gid_y = get_global_id(1);
	int lid_x = get_local_id(0);
	int lid_y = get_local_id(1);

	// "center" index of the kernel. This is also the offset into the input image.
	int2 kernel_pivot_idx = kernel_pivot_index(kernel_size);
	int2 kernel_start_idx = -kernel_pivot_idx;
	int2 kernel_end_idx = kernel_size - kernel_pivot_idx - 1;
	float2 input_pivot = (float2)(
		(float)(kernel_pivot_idx.x + gid_x) + 0.5f,
		(float)(kernel_pivot_idx.y + gid_y) + 0.5f 
	);
	float2 kernel_pivot = (float2)(
		(float)kernel_pivot_idx.x + 0.5f,
		(float)kernel_pivot_idx.y + 0.5f
	);
	
	float sqdiff = 0.0f;
	// for each layer
	for(int layer = 0; layer < get_image_array_size(input_tex); ++layer)
	{
		// iterate over kernel area
		for(int dy = kernel_start_idx.y; dy != kernel_end_idx.y; ++dy)
		{
			for(int dx = kernel_start_idx.x; dx != kernel_end_idx.x; ++dx)
			{
				float2 cdelta = (float2)((float)dx, (float)dy);
				float4 image_coord = (float4)(
					rotate_sample(cdelta + input_pivot, input_pivot, rotation_sincos),
					(float)layer,
					0.0f
				);
				// TODO: If kernel is small, store kernel values in local memory before doing this loop!
				float4 kernel_coord = (float4)(
					kernel_pivot + cdelta,
					(float)layer,
					0.0f
				);
				float diff = read_imagef(input_tex, input_sampler, image_coord).x - read_imagef(kernel_tex, input_sampler, kernel_coord).x;
				sqdiff += diff * diff;
			}
		}
	}

	// write result
	write_imagef(response_tex, (int2)(gid_x, gid_y), (float4)(sqdiff, 0.0f, 0.0f, 0.0f));
}

__kernel void sqdiff_naive_kernel_mask(
	__read_only image2d_array_t input_tex,
	__read_only image2d_array_t kernel_tex,
	__read_only image2d_t kernel_mask,
	__write_only image2d_t response_tex,
	int2 input_size,
	int2 kernel_size,
	float2 rotation_sincos)
{
	int gid_x = get_global_id(0);
	int gid_y = get_global_id(1);
	int lid_x = get_local_id(0);
	int lid_y = get_local_id(1);

	// "center" index of the kernel. This is also the offset into the input image.
	int2 kernel_pivot_idx = kernel_pivot_index(kernel_size);
	int2 kernel_start_idx = -kernel_pivot_idx;
	int2 kernel_end_idx = kernel_size - kernel_pivot_idx - 1;
	float2 input_pivot = (float2)(
		(float)(kernel_pivot_idx.x + gid_x) + 0.5f,
		(float)(kernel_pivot_idx.y + gid_y) + 0.5f 
	);
	float2 kernel_pivot = (float2)(
		(float)kernel_pivot_idx.x + 0.5f,
		(float)kernel_pivot_idx.y + 0.5f
	);
	
	float sqdiff = 0.0f;
	// for each layer
	for(int layer = 0; layer < get_image_array_size(input_tex); ++layer)
	{
		// iterate over kernel area
		for(int dy = kernel_start_idx.y; dy != kernel_end_idx.y; ++dy)
		{
			for(int dx = kernel_start_idx.x; dx != kernel_end_idx.x; ++dx)
			{
				float2 cdelta = (float2)((float)dx, (float)dy);
				float4 image_coord = (float4)(
					rotate_sample(cdelta + input_pivot, input_pivot, rotation_sincos),
					(float)layer,
					0.0f
				);
				// TODO: If kernel is small, store kernel values in local memory before doing this loop!
				float4 kernel_coord = (float4)(
					kernel_pivot + cdelta,
					(float)layer,
					0.0f
				);
				float diff = read_imagef(input_tex, input_sampler, image_coord).x - read_imagef(kernel_tex, input_sampler, kernel_coord).x;
				sqdiff += diff * diff;
			}
		}
	}

	// write result
	write_imagef(response_tex, (int2)(gid_x, gid_y), (float4)(sqdiff, 0.0f, 0.0f, 0.0f));
}

__kernel void sqdiff_naive_no_mask(
	__read_only image2d_array_t input_tex,
	__read_only image2d_array_t kernel_tex,
	__write_only image2d_t response_tex,
	int2 input_size,
	int2 kernel_size,
	float2 rotation_sincos)
{
	int gid_x = get_global_id(0);
	int gid_y = get_global_id(1);
	int lid_x = get_local_id(0);
	int lid_y = get_local_id(1);

	// "center" index of the kernel. This is also the offset into the input image.
	int2 kernel_pivot_idx = kernel_pivot_index(kernel_size);
	int2 kernel_start_idx = -kernel_pivot_idx;
	int2 kernel_end_idx = kernel_size - kernel_pivot_idx - 1;
	float2 input_pivot = (float2)(
		(float)(kernel_pivot_idx.x + gid_x) + 0.5f,
		(float)(kernel_pivot_idx.y + gid_y) + 0.5f 
	);
	float2 kernel_pivot = (float2)(
		(float)kernel_pivot_idx.x + 0.5f,
		(float)kernel_pivot_idx.y + 0.5f
	);
	
	float sqdiff = 0.0f;
	// for each layer
	for(int layer = 0; layer < get_image_array_size(input_tex); ++layer)
	{
		// iterate over kernel area
		for(int dy = kernel_start_idx.y; dy != kernel_end_idx.y; ++dy)
		{
			for(int dx = kernel_start_idx.x; dx != kernel_end_idx.x; ++dx)
			{
				float2 cdelta = (float2)((float)dx, (float)dy);
				float4 image_coord = (float4)(
					rotate_sample(cdelta + input_pivot, input_pivot, rotation_sincos),
					(float)layer,
					0.0f
				);
				// TODO: If kernel is small, store kernel values in local memory before doing this loop!
				float4 kernel_coord = (float4)(
					kernel_pivot + cdelta,
					(float)layer,
					0.0f
				);
				float diff = read_imagef(input_tex, input_sampler, image_coord).x - read_imagef(kernel_tex, input_sampler, kernel_coord).x;
				sqdiff += diff * diff;
			}
		}
	}

	// write result
	write_imagef(response_tex, (int2)(gid_x, gid_y), (float4)(sqdiff, 0.0f, 0.0f, 0.0f));
}