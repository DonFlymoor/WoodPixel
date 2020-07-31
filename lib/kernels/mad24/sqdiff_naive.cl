#define MASK_THRESHOLD 1e-6f
#define PIXEL_CENTER_OFFSET 0.5f
// sampler (maybe a constant border color would be better? don't know yet..)
const sampler_t input_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
const sampler_t kernel_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void sqdiff_naive_masked(
	__read_only image2d_t input_tex,
	__read_only image2d_t kernel_tex,
	__read_only image2d_t kernel_mask,
	__write_only image2d_t response_tex,
	int2 input_size,
	int2 kernel_size,
	int2 kernel_anchor,
	int2 input_piv,
	float2 rotation_sincos)
{
	const int gid_x = get_global_id(0);
	const int gid_y = get_global_id(1);

	// "center" index of the kernel. This is also the offset into the input image.
	const int2 kernel_start_idx = -kernel_anchor;
	const int2 kernel_end_idx = kernel_size - kernel_anchor - 1;
	
	const float2 input_pivot = (float2)(
		(float)(input_piv.x + gid_x) + PIXEL_CENTER_OFFSET,
		(float)(input_piv.y + gid_y) + PIXEL_CENTER_OFFSET
	);
	const float2 kernel_pivot = (float2)(
		(float)kernel_anchor.x + PIXEL_CENTER_OFFSET,
		(float)kernel_anchor.y + PIXEL_CENTER_OFFSET
	);
	
	float sqdiff = 0.0f;
	float4 diff;
	float2 cdelta = (float2)(0.0f);
	float2 image_coord;
	float2 kernel_coord;
	
	// iterate over kernel area

	// columns of the rotation matrix
	const float2 r0 = (float2)(rotation_sincos.y, rotation_sincos.x);
	const float2 r1 = (float2)(-rotation_sincos.x, rotation_sincos.y);

	for(int dy = kernel_start_idx.y; dy != kernel_end_idx.y; ++dy)
	{
		for(int dx = kernel_start_idx.x; dx != kernel_end_idx.x; ++dx)
		{
			cdelta = (float2)((float)dx, (float)dy);
			
			// calculate image coord (applies rotation around current texel!)
			// image_coord.x = rotation_sincos.y * cdelta.x - rotation_sincos.x * cdelta.y + input_pivot.x;
			// image_coord.y = rotation_sincos.x * cdelta.x + rotation_sincos.y * cdelta.y + input_pivot.y;
			image_coord = cdelta.x * r0 + cdelta.y * r1 + input_pivot;

			// calculate kernel coord
			kernel_coord = kernel_pivot + cdelta;

			// squared difference
			float mask_val = read_imagef(kernel_mask, kernel_sampler, kernel_coord).x;
			if(mask_val > MASK_THRESHOLD)
			{
				diff = (read_imagef(input_tex, input_sampler, image_coord) - read_imagef(kernel_tex, kernel_sampler, kernel_coord));
				sqdiff += dot(diff, diff);
			}
		}
	}
	
	// write result
	write_imagef(response_tex, (int2)(gid_x, gid_y), (float4)(sqdiff, 0.0f, 0.0f, 0.0f));
}

__kernel void sqdiff_naive_masked_nth_pass(
	__read_only image2d_t input_tex,
	__read_only image2d_t kernel_tex,
	__read_only image2d_t kernel_mask,
	__read_only image2d_t prev_response_tex,
	__write_only image2d_t response_tex,
	int2 input_size,
	int2 kernel_size,
	int2 kernel_anchor,
	int2 input_piv,
	float2 rotation_sincos)
{
	const int gid_x = get_global_id(0);
	const int gid_y = get_global_id(1);

	// "center" index of the kernel. This is also the offset into the input image.
	const int2 kernel_start_idx = -kernel_anchor;
	const int2 kernel_end_idx = kernel_size - kernel_anchor - 1;
	
	const float2 input_pivot = (float2)(
		(float)(input_piv.x + gid_x) + PIXEL_CENTER_OFFSET,
		(float)(input_piv.y + gid_y) + PIXEL_CENTER_OFFSET
	);
	const float2 kernel_pivot = (float2)(
		(float)kernel_anchor.x + PIXEL_CENTER_OFFSET,
		(float)kernel_anchor.y + PIXEL_CENTER_OFFSET
	);
	
	float sqdiff = 0.0f;
	float4 diff;
	float2 cdelta = (float2)(0.0f);
	float2 image_coord;
	float2 kernel_coord;
	
	// iterate over kernel area

	// columns of the rotation matrix
	const float2 r0 = (float2)(rotation_sincos.y, rotation_sincos.x);
	const float2 r1 = (float2)(-rotation_sincos.x, rotation_sincos.y);

	for(int dy = kernel_start_idx.y; dy != kernel_end_idx.y; ++dy)
	{
		for(int dx = kernel_start_idx.x; dx != kernel_end_idx.x; ++dx)
		{
			cdelta = (float2)((float)dx, (float)dy);
			
			// calculate image coord (applies rotation around current texel!)
			// image_coord.x = rotation_sincos.y * cdelta.x - rotation_sincos.x * cdelta.y + input_pivot.x;
			// image_coord.y = rotation_sincos.x * cdelta.x + rotation_sincos.y * cdelta.y + input_pivot.y;
			image_coord = cdelta.x * r0 + cdelta.y * r1 + input_pivot;

			// calculate kernel coord
			kernel_coord = kernel_pivot + cdelta;

			// squared difference
			float mask_val = read_imagef(kernel_mask, kernel_sampler, kernel_coord).x;
			if(mask_val > MASK_THRESHOLD)
			{
				diff = (read_imagef(input_tex, input_sampler, image_coord) - read_imagef(kernel_tex, kernel_sampler, kernel_coord));
				sqdiff += dot(diff, diff);
			}
		}
	}
	
	// write result
	int2 out_coord = (int2)(gid_x, gid_y);
	write_imagef(response_tex, out_coord, (float4)(sqdiff, 0.0f, 0.0f, 0.0f) + read_imagef(prev_response_tex, kernel_sampler, out_coord));
}

__kernel void sqdiff_naive(
	__read_only image2d_t input_tex,
	__read_only image2d_t kernel_tex,
	__write_only image2d_t response_tex,
	int2 input_size,
	int2 kernel_size,
	int2 kernel_anchor,
	int2 input_piv,
	float2 rotation_sincos)
{
	const int gid_x = get_global_id(0);
	const int gid_y = get_global_id(1);

	// "center" index of the kernel. This is also the offset into the input image.
	const int2 kernel_start_idx = -kernel_anchor;
	const int2 kernel_end_idx = kernel_size - kernel_anchor - 1;
	
	const float2 input_pivot = (float2)(
		(float)(input_piv.x + gid_x) + PIXEL_CENTER_OFFSET,
		(float)(input_piv.y + gid_y) + PIXEL_CENTER_OFFSET
	);
	const float2 kernel_pivot = (float2)(
		(float)kernel_anchor.x + PIXEL_CENTER_OFFSET,
		(float)kernel_anchor.y + PIXEL_CENTER_OFFSET
	);
	
	float sqdiff = 0.0f;
	float4 diff;
	float2 cdelta = (float2)(0.0f);
	float2 image_coord;
	float2 kernel_coord;
	
	// iterate over kernel area

	// columns of the rotation matrix
	const float2 r0 = (float2)(rotation_sincos.y, rotation_sincos.x);
	const float2 r1 = (float2)(-rotation_sincos.x, rotation_sincos.y);

	for(int dy = kernel_start_idx.y; dy != kernel_end_idx.y; ++dy)
	{
		for(int dx = kernel_start_idx.x; dx != kernel_end_idx.x; ++dx)
		{
			cdelta = (float2)((float)dx, (float)dy);
			
			// calculate image coord (applies rotation around current texel!)
			// image_coord.x = rotation_sincos.y * cdelta.x - rotation_sincos.x * cdelta.y + input_pivot.x;
			// image_coord.y = rotation_sincos.x * cdelta.x + rotation_sincos.y * cdelta.y + input_pivot.y;
			image_coord = cdelta.x * r0 + cdelta.y * r1 + input_pivot;

			// calculate kernel coord
			kernel_coord = kernel_pivot + cdelta;

			// squared difference
			diff = (read_imagef(input_tex, input_sampler, image_coord) - read_imagef(kernel_tex, kernel_sampler, kernel_coord));
			sqdiff += dot(diff, diff);
		}
	}

	// write result
	write_imagef(response_tex, (int2)(gid_x, gid_y), (float4)(sqdiff, 0.0f, 0.0f, 0.0f));	
}

__kernel void sqdiff_naive_nth_pass(
	__read_only image2d_t input_tex,
	__read_only image2d_t kernel_tex,
	__read_only image2d_t prev_response_tex,
	__write_only image2d_t response_tex,
	int2 input_size,
	int2 kernel_size,
	int2 kernel_anchor,
	int2 input_piv,
	float2 rotation_sincos)
{
	const int gid_x = get_global_id(0);
	const int gid_y = get_global_id(1);

	// "center" index of the kernel. This is also the offset into the input image.
	const int2 kernel_start_idx = -kernel_anchor;
	const int2 kernel_end_idx = kernel_size - kernel_anchor - 1;
	
	const float2 input_pivot = (float2)(
		(float)(input_piv.x + gid_x) + PIXEL_CENTER_OFFSET,
		(float)(input_piv.y + gid_y) + PIXEL_CENTER_OFFSET
	);
	const float2 kernel_pivot = (float2)(
		(float)kernel_anchor.x + PIXEL_CENTER_OFFSET,
		(float)kernel_anchor.y + PIXEL_CENTER_OFFSET
	);
	
	float sqdiff = 0.0f;
	float4 diff;
	float2 cdelta = (float2)(0.0f);
	float2 image_coord;
	float2 kernel_coord;
	
	// iterate over kernel area

	// columns of the rotation matrix
	const float2 r0 = (float2)(rotation_sincos.y, rotation_sincos.x);
	const float2 r1 = (float2)(-rotation_sincos.x, rotation_sincos.y);

	for(int dy = kernel_start_idx.y; dy != kernel_end_idx.y; ++dy)
	{
		for(int dx = kernel_start_idx.x; dx != kernel_end_idx.x; ++dx)
		{
			cdelta = (float2)((float)dx, (float)dy);
			
			// calculate image coord (applies rotation around current texel!)
			// image_coord.x = rotation_sincos.y * cdelta.x - rotation_sincos.x * cdelta.y + input_pivot.x;
			// image_coord.y = rotation_sincos.x * cdelta.x + rotation_sincos.y * cdelta.y + input_pivot.y;
			image_coord = cdelta.x * r0 + cdelta.y * r1 + input_pivot;

			// calculate kernel coord
			kernel_coord = kernel_pivot + cdelta;

			// squared difference
			diff = (read_imagef(input_tex, input_sampler, image_coord) - read_imagef(kernel_tex, kernel_sampler, kernel_coord));
			sqdiff += dot(diff, diff);
		}
	}

	// write result
	int2 out_coord = (int2)(gid_x, gid_y);
	write_imagef(response_tex, out_coord, (float4)(sqdiff, 0.0f, 0.0f, 0.0f) + read_imagef(prev_response_tex, kernel_sampler, out_coord));
}