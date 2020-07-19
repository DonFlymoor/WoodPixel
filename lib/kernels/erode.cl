const sampler_t mask_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
// something is off here
__kernel void erode(
	__read_only image2d_t input_tex,
	__read_only image2d_t kernel_tex,
	__write_only image2d_t response_tex,
	int2 input_size,
	int2 kernel_size,
	int2 kernel_anchor,
	float2 rotation_sincos)
{
	const int gid_x = get_global_id(0);
	const int gid_y = get_global_id(1);

	// "center" index of the kernel. This is also the offset into the input image.
	const int2 kernel_start_idx = -kernel_anchor;
	const int2 kernel_end_idx = kernel_size - kernel_anchor - 1;
	
	const float2 input_pivot = (float2)(
		(float)(kernel_anchor.x + gid_x) + 0.5f,
		(float)(kernel_anchor.y + gid_y) + 0.5f
	);
	const float2 kernel_pivot = (float2)(
		(float)kernel_anchor.x + 0.5f,
		(float)kernel_anchor.y + 0.5f
	);
	
	float minval = 1.0f;
	float2 cdelta = (float2)(0.0f);
	float2 image_coord;
	float2 kernel_coord;
	
	// iterate over kernel area
	for(int dy = kernel_start_idx.y; dy != kernel_end_idx.y; ++dy)
	{
		for(int dx = kernel_start_idx.x; dx != kernel_end_idx.x; ++dx)
		{
			cdelta = (float2)((float)dx, (float)dy);
			
			// calculate image coord (applies rotation around current texel!)
			image_coord.x = rotation_sincos.y * cdelta.x - rotation_sincos.x * cdelta.y + input_pivot.x;
			image_coord.y = rotation_sincos.x * cdelta.x + rotation_sincos.y * cdelta.y + input_pivot.y;

			// calculate kernel coord
			kernel_coord = kernel_pivot + cdelta;

			float kernel_val = read_imagef(kernel_tex, mask_sampler, kernel_coord).x; // 1.0 or 0.0
			if(kernel_val > 0.5f) // no divergence because kernel values are the same for every thread
			{
				float image_val = step(0.5f, read_imagef(input_tex, mask_sampler, image_coord).x); // 1.0 or 0.0
				minval = min(minval, image_val);
			}			
		}
	}
	// write result
	write_imagef(response_tex, (int2)(gid_x, gid_y), (float4)(minval, 0.0f, 0.0f, 0.0f));
}

__kernel void erode_constant(
	__read_only image2d_t input_tex,
	__constant float* kernel_tex,
	__write_only image2d_t response_tex,
	int2 input_size,
	int2 kernel_size,
	int2 kernel_anchor,
	float2 rotation_sincos)
{
	const int gid_x = get_global_id(0);
	const int gid_y = get_global_id(1);

	// "center" index of the kernel. This is also the offset into the input image.
	const int2 kernel_start_idx = -kernel_anchor;
	const int2 kernel_end_idx = kernel_size - kernel_anchor - 1;
	
	const float2 input_pivot = (float2)(
		(float)(kernel_anchor.x + gid_x) + 0.5f,
		(float)(kernel_anchor.y + gid_y) + 0.5f
	);	
	float minval = 1.0f;
	float2 cdelta = (float2)(0.0f);
	float2 image_coord;	
	// iterate over kernel area
	int kernel_pix_idx;
	for(int dy = kernel_start_idx.y; dy != kernel_end_idx.y; ++dy)
	{
		for(int dx = kernel_start_idx.x; dx != kernel_end_idx.x; ++dx)
		{
			cdelta = (float2)((float)dx, (float)dy);
			
			// calculate image coord (applies rotation around current texel!)
			image_coord.x = rotation_sincos.y * cdelta.x - rotation_sincos.x * cdelta.y + input_pivot.x;
			image_coord.y = rotation_sincos.x * cdelta.x + rotation_sincos.y * cdelta.y + input_pivot.y;

			// squared difference
			kernel_pix_idx = (kernel_anchor.y + dy) * kernel_size.x + (kernel_anchor.x + dx);
			float kernel_val = kernel_tex[kernel_pix_idx];
			if(kernel_val > 0.5f) // no divergence because kernel values are the same for every thread
			{
				float image_val = step(0.5f, read_imagef(input_tex, mask_sampler, image_coord).x);			
				minval = min(minval, image_val);
			}
		}
	}
	// write result
	write_imagef(response_tex, (int2)(gid_x, gid_y), (float4)(minval, 0.0f, 0.0f, 0.0f));
}