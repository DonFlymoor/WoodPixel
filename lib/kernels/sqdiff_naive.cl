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
	float2 rotation_sincos)
{
	const int gid_x = get_global_id(0);
	const int gid_y = get_global_id(1);

	// "center" index of the kernel. This is also the offset into the input image.
	const int2 kernel_pivot_idx = (kernel_size - 1) / 2;
	const int2 kernel_start_idx = -kernel_pivot_idx;
	const int2 kernel_end_idx = kernel_size - kernel_pivot_idx - 1;
	
	const float2 input_pivot = (float2)(
		(float)(kernel_pivot_idx.x + gid_x) + 0.5f,
		(float)(kernel_pivot_idx.y + gid_y) + 0.5f
	);
	const float2 kernel_pivot = (float2)(
		(float)kernel_pivot_idx.x + 0.5f,
		(float)kernel_pivot_idx.y + 0.5f
	);
	
	float sqdiff = 0.0f;
	float4 diff;
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
			image_coord = cdelta;
			image_coord.x = rotation_sincos.y * image_coord.x - rotation_sincos.x * image_coord.y;
			image_coord.y = rotation_sincos.x * image_coord.x + rotation_sincos.y * image_coord.y;
			image_coord += input_pivot;

			// calculate kernel coord
			kernel_coord = kernel_pivot + cdelta;

			// squared difference
			diff = (read_imagef(input_tex, input_sampler, image_coord) - read_imagef(kernel_tex, kernel_sampler, kernel_coord));
			sqdiff += dot(diff, diff) * read_imagef(kernel_mask, kernel_sampler, kernel_coord).x;
		}
	}
	
	// write result
	write_imagef(response_tex, (int2)(gid_x, gid_y), (float4)(sqdiff, 0.0f, 0.0f, 0.0f));
	// debug output
	// if(gid_x < kernel_size.x && gid_y < kernel_size.y)
	// {
	// 	write_imagef(response_tex, (int2)(gid_x, gid_y), (float4)(read_imagef(kernel_tex, input_sampler, (float2)((float)gid_x + 0.5f, (float)gid_y + 0.5f)).x, 0.0f, 0.0f, 0.0f));
	// }
	// else
	// {
	// 	write_imagef(response_tex, (int2)(gid_x, gid_y), (float4)(0.0f, 0.0f, 0.0f, 0.0f));
	// }
}

__kernel void sqdiff_naive_masked_nth_pass(
	__read_only image2d_t input_tex,
	__read_only image2d_t kernel_tex,
	__read_only image2d_t kernel_mask,
	__read_only image2d_t prev_response_tex,
	__write_only image2d_t response_tex,
	int2 input_size,
	int2 kernel_size,
	float2 rotation_sincos)
{
	const int gid_x = get_global_id(0);
	const int gid_y = get_global_id(1);

	// "center" index of the kernel. This is also the offset into the input image.
	const int2 kernel_pivot_idx = (kernel_size - 1) / 2;
	const int2 kernel_start_idx = -kernel_pivot_idx;
	const int2 kernel_end_idx = kernel_size - kernel_pivot_idx - 1;
	
	const float2 input_pivot = (float2)(
		(float)(kernel_pivot_idx.x + gid_x) + 0.5f,
		(float)(kernel_pivot_idx.y + gid_y) + 0.5f
	);
	const float2 kernel_pivot = (float2)(
		(float)kernel_pivot_idx.x + 0.5f,
		(float)kernel_pivot_idx.y + 0.5f
	);
	
	float sqdiff = 0.0f;
	float4 diff;
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
			image_coord = cdelta;
			image_coord.x = rotation_sincos.y * image_coord.x - rotation_sincos.x * image_coord.y;
			image_coord.y = rotation_sincos.x * image_coord.x + rotation_sincos.y * image_coord.y;
			image_coord += input_pivot;

			// calculate kernel coord
			kernel_coord = kernel_pivot + cdelta;

			// squared difference
			diff = (read_imagef(input_tex, input_sampler, image_coord) - read_imagef(kernel_tex, kernel_sampler, kernel_coord));
			sqdiff += dot(diff, diff) * read_imagef(kernel_mask, kernel_sampler, kernel_coord).x;
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
	float2 rotation_sincos)
{
	const int gid_x = get_global_id(0);
	const int gid_y = get_global_id(1);

	// "center" index of the kernel. This is also the offset into the input image.
	const int2 kernel_pivot_idx = (kernel_size - 1) / 2;
	const int2 kernel_start_idx = -kernel_pivot_idx;
	const int2 kernel_end_idx = kernel_size - kernel_pivot_idx - 1;
	
	const float2 input_pivot = (float2)(
		(float)(kernel_pivot_idx.x + gid_x) + 0.5f,
		(float)(kernel_pivot_idx.y + gid_y) + 0.5f
	);
	const float2 kernel_pivot = (float2)(
		(float)kernel_pivot_idx.x + 0.5f,
		(float)kernel_pivot_idx.y + 0.5f
	);
	
	float sqdiff = 0.0f;
	float4 diff;
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
			image_coord = cdelta;
			image_coord.x = rotation_sincos.y * image_coord.x - rotation_sincos.x * image_coord.y;
			image_coord.y = rotation_sincos.x * image_coord.x + rotation_sincos.y * image_coord.y;
			image_coord += input_pivot;

			// calculate kernel coord
			kernel_coord = kernel_pivot + cdelta;

			// squared difference
			diff = (read_imagef(input_tex, input_sampler, image_coord) - read_imagef(kernel_tex, kernel_sampler, kernel_coord));
			sqdiff += dot(diff, diff);
		}
	}

	// write result
	write_imagef(response_tex, (int2)(gid_x, gid_y), (float4)(sqdiff, 0.0f, 0.0f, 0.0f));
	// debug output
	// if(gid_x < kernel_size.x && gid_y < kernel_size.y)
	// {
	// 	write_imagef(response_tex, (int2)(gid_x, gid_y), (float4)(read_imagef(kernel_tex, input_sampler, (float2)((float)gid_x + 0.5f, (float)gid_y + 0.5f)).x, 0.0f, 0.0f, 0.0f));
	// }
	// else
	// {
	// 	write_imagef(response_tex, (int2)(gid_x, gid_y), (float4)(0.0f, 0.0f, 0.0f, 0.0f));
	// }
}

__kernel void sqdiff_naive_nth_pass(
	__read_only image2d_t input_tex,
	__read_only image2d_t kernel_tex,
	__read_only image2d_t prev_response_tex,
	__write_only image2d_t response_tex,
	int2 input_size,
	int2 kernel_size,
	float2 rotation_sincos)
{
	const int gid_x = get_global_id(0);
	const int gid_y = get_global_id(1);

	// "center" index of the kernel. This is also the offset into the input image.
	const int2 kernel_pivot_idx = (kernel_size - 1) / 2;
	const int2 kernel_start_idx = -kernel_pivot_idx;
	const int2 kernel_end_idx = kernel_size - kernel_pivot_idx - 1;
	
	const float2 input_pivot = (float2)(
		(float)(kernel_pivot_idx.x + gid_x) + 0.5f,
		(float)(kernel_pivot_idx.y + gid_y) + 0.5f
	);
	const float2 kernel_pivot = (float2)(
		(float)kernel_pivot_idx.x + 0.5f,
		(float)kernel_pivot_idx.y + 0.5f
	);
	
	float sqdiff = 0.0f;
	float4 diff;
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
			image_coord = cdelta;
			image_coord.x = rotation_sincos.y * image_coord.x - rotation_sincos.x * image_coord.y;
			image_coord.y = rotation_sincos.x * image_coord.x + rotation_sincos.y * image_coord.y;
			image_coord += input_pivot;

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