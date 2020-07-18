// sampler (maybe a constant border color would be better? don't know yet..)
const sampler_t input_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
const sampler_t kernel_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void sqdiff_constant(
	__read_only image2d_t input_tex,
	__local float4* image_local_buffer,
	__constant float4* kernel_tex,
	__write_only image2d_t response_tex,
	int2 input_size,
	int2 kernel_size,
	int4 kernel_overlaps,
	float2 rotation_sincos)
{
	const int gid_x = get_global_id(0);
	const int gid_y = get_global_id(1);
	const int lid_x = get_local_id(1);
	const int lid_y = get_local_id(1);
	const int ls_x = get_local_size(0);
	const int ls_y = get_local_size(1);
	const int lid = lid_y * ls_x + lid_x;

	// "center" index of the kernel. This is also the offset into the input image.
	const int2 kernel_pivot_idx = (kernel_size - 1) / 2;
	const int2 kernel_start_idx = -kernel_pivot_idx;
	const int2 kernel_end_idx = kernel_size - kernel_pivot_idx - 1;
	
	const float2 input_pivot = (float2)(
		(float)(kernel_pivot_idx.x + gid_x) + 0.5f,
		(float)(kernel_pivot_idx.y + gid_y) + 0.5f
	);

	// load data into local memory
	// overlap sizes
	const int hl = kernel_overlaps.x;	// overlap left
	const int hr = kernel_overlaps.y;	// overlap right
	const int ht = kernel_overlaps.z;	// operlap top
	const int hb = kernel_overlaps.w;	// overlap bottom
	const int local_buffer_width = ls_x + hl + hr;
	const int local_buffer_height = ls_y + ht + hb;
	// for sampling the local buffer in the loop below
	const float2 local_mem_pivot = (float2)((float)(hl + lid_x) + 0.5f, (float)(ht + lid_y) + 0.5f);
	// load center data
	image_local_buffer[(ht + lid_y) * local_buffer_width + (hl + lid_x)] = read_imagef(input_tex, input_sampler, input_pivot);
	// load overlap texels
	// vertical edges
	if(lid_x < hl)
		image_local_buffer[(ht + lid_y) * local_buffer_width + (lid_x)] = read_imagef(input_tex, input_sampler, input_pivot - (float2)((float)hl, 0.0f));
	else if(lid_x >= (ls_x - hr))
		image_local_buffer[(ht + lid_y) * local_buffer_width + (hl + lid_x + hr)] = read_imagef(input_tex, input_sampler, input_pivot + (float2)((float)hr, 0.0f));
	// horizontal edges
	if(lid_y < ht)
		image_local_buffer[(lid_y) * local_buffer_width + (hl + lid_x)] = read_imagef(input_tex, input_sampler, input_pivot - (float2)(0.0f, (float)ht));
	else if(lid_y >= (ls_y - hb))
		image_local_buffer[(ht + lid_y + hb) * local_buffer_width + (hl + lid_x)] = read_imagef(input_tex, input_sampler, input_pivot + (float2)(0.0f, (float)hb));
	// four corners
	if((lid_x < hl) && (lid_y < ht)) // upper left corner
		image_local_buffer[(lid_y) * local_buffer_width + (lid_x)] = read_imagef(input_tex, input_sampler, input_pivot - (float2)((float)hl, (float)ht));
	else if((lid_x >= (ls_x - hr)) && (lid_y < ht)) // upper right corner
		image_local_buffer[(lid_y) * local_buffer_width + (hl + lid_x + hr)] = read_imagef(input_tex, input_sampler, input_pivot + (float2)((float)hr, (float)-ht));
	else if((lid_x < hl) && (lid_y >= (ls_y - hb))) // lower left corner
		image_local_buffer[(ht + lid_y + hb) * local_buffer_width + (lid_x)] = read_imagef(input_tex, input_sampler, input_pivot + (float2)((float)-hl, (float)hb));
	else if((lid_x >= (ls_x - hr)) && (lid_y >= (ls_y - hb))) // lower right corner
		image_local_buffer[(ht + lid_y + hb) * local_buffer_width + (hl + lid_x + hr)] = read_imagef(input_tex, input_sampler, input_pivot + (float2)((float)hr, (float)hb));

	// synchronize all local buffer writes
	barrier(CLK_LOCAL_MEM_FENCE);

	float sqdiff = 0.0f;
	float4 diff;
	float2 cdelta = (float2)(0.0f);
	float2 local_mem_coord;
	int local_buffer_idx;
	
	// iterate over kernel area
	int kernel_pix_idx;
	for(int dy = kernel_start_idx.y; dy != kernel_end_idx.y; ++dy)
	{
		for(int dx = kernel_start_idx.x; dx != kernel_end_idx.x; ++dx)
		{
			cdelta = (float2)((float)dx, (float)dy);
			
			// calculate image coord (applies rotation around current texel!)
			local_mem_coord.x = rotation_sincos.y * cdelta.x - rotation_sincos.x * cdelta.y + local_mem_pivot.x;
			local_mem_coord.y = rotation_sincos.x * cdelta.x + rotation_sincos.y * cdelta.y + local_mem_pivot.y;
			local_buffer_idx = (int)floor(local_mem_coord.y) * local_buffer_width + (int)floor(local_mem_coord.x);
			// squared difference
			kernel_pix_idx = (kernel_pivot_idx.y + dy) * kernel_size.x + (kernel_pivot_idx.x + dx);			
			diff = image_local_buffer[local_buffer_idx] - kernel_tex[kernel_pix_idx];
			sqdiff += dot(diff, diff);			
		}
	}
	
	// write result
	write_imagef(response_tex, (int2)(gid_x, gid_y), (float4)(sqdiff, 0.0f, 0.0f, 0.0f));
}

__kernel void sqdiff_constant_nth_pass(
	__read_only image2d_t input_tex,
	__local float4* image_local_buffer,
	__constant float4* kernel_tex,
	__read_only image2d_t prev_response_tex,
	__write_only image2d_t response_tex,
	int2 input_size,
	int2 kernel_size,
	int4 kernel_overlaps,
	float2 rotation_sincos,
	int kernel_offset)
{
	const int gid_x = get_global_id(0);
	const int gid_y = get_global_id(1);
	const int lid_x = get_local_id(1);
	const int lid_y = get_local_id(1);
	const int ls_x = get_local_size(0);
	const int ls_y = get_local_size(1);
	const int lid = lid_y * ls_x + lid_x;

	// "center" index of the kernel. This is also the offset into the input image.
	const int2 kernel_pivot_idx = (kernel_size - 1) / 2;
	const int2 kernel_start_idx = -kernel_pivot_idx;
	const int2 kernel_end_idx = kernel_size - kernel_pivot_idx - 1;
	
	const float2 input_pivot = (float2)(
		(float)(kernel_pivot_idx.x + gid_x) + 0.5f,
		(float)(kernel_pivot_idx.y + gid_y) + 0.5f
	);

	// load data into local memory
	// overlap sizes
	const int hl = kernel_overlaps.x;	// overlap left
	const int hr = kernel_overlaps.y;	// overlap right
	const int ht = kernel_overlaps.z;	// operlap top
	const int hb = kernel_overlaps.w;	// overlap bottom
	const int local_buffer_width = ls_x + hl + hr;
	const int local_buffer_height = ls_y + ht + hb;
	// for sampling the local buffer in the loop below
	const float2 local_mem_pivot = (float2)((float)(hl + lid_x) + 0.5f, (float)(ht + lid_y) + 0.5f);
	// load center data
	image_local_buffer[(ht + lid_y) * local_buffer_width + (hl + lid_x)] = read_imagef(input_tex, input_sampler, input_pivot);
	// load overlap texels
	// vertical edges
	if(lid_x < hl)
		image_local_buffer[(ht + lid_y) * local_buffer_width + (lid_x)] = read_imagef(input_tex, input_sampler, input_pivot - (float2)((float)hl, 0.0f));
	else if(lid_x >= (ls_x - hr))
		image_local_buffer[(ht + lid_y) * local_buffer_width + (hl + lid_x + hr)] = read_imagef(input_tex, input_sampler, input_pivot + (float2)((float)hr, 0.0f));
	// horizontal edges
	if(lid_y < ht)
		image_local_buffer[(lid_y) * local_buffer_width + (hl + lid_x)] = read_imagef(input_tex, input_sampler, input_pivot - (float2)(0.0f, (float)ht));
	else if(lid_y >= (ls_y - hb))
		image_local_buffer[(ht + lid_y + hb) * local_buffer_width + (hl + lid_x)] = read_imagef(input_tex, input_sampler, input_pivot + (float2)(0.0f, (float)hb));
	// four corners
	if((lid_x < hl) && (lid_y < ht)) // upper left corner
		image_local_buffer[(lid_y) * local_buffer_width + (lid_x)] = read_imagef(input_tex, input_sampler, input_pivot - (float2)((float)hl, (float)ht));
	else if((lid_x >= (ls_x - hr)) && (lid_y < ht)) // upper right corner
		image_local_buffer[(lid_y) * local_buffer_width + (hl + lid_x + hr)] = read_imagef(input_tex, input_sampler, input_pivot + (float2)((float)hr, (float)-ht));
	else if((lid_x < hl) && (lid_y >= (ls_y - hb))) // lower left corner
		image_local_buffer[(ht + lid_y + hb) * local_buffer_width + (lid_x)] = read_imagef(input_tex, input_sampler, input_pivot + (float2)((float)-hl, (float)hb));
	else if((lid_x >= (ls_x - hr)) && (lid_y >= (ls_y - hb))) // lower right corner
		image_local_buffer[(ht + lid_y + hb) * local_buffer_width + (hl + lid_x + hr)] = read_imagef(input_tex, input_sampler, input_pivot + (float2)((float)hr, (float)hb));

	// synchronize all local buffer writes
	barrier(CLK_LOCAL_MEM_FENCE);

	float sqdiff = 0.0f;
	float4 diff;
	float2 cdelta = (float2)(0.0f);
	float2 local_mem_coord;
	int local_buffer_idx;
	
	// iterate over kernel area
	int kernel_pix_idx;
	for(int dy = kernel_start_idx.y; dy != kernel_end_idx.y; ++dy)
	{
		for(int dx = kernel_start_idx.x; dx != kernel_end_idx.x; ++dx)
		{
			cdelta = (float2)((float)dx, (float)dy);
			
			// calculate image coord (applies rotation around current texel!)
			local_mem_coord.x = rotation_sincos.y * cdelta.x - rotation_sincos.x * cdelta.y + local_mem_pivot.x;
			local_mem_coord.y = rotation_sincos.x * cdelta.x + rotation_sincos.y * cdelta.y + local_mem_pivot.y;
			local_buffer_idx = (int)floor(local_mem_coord.y) * local_buffer_width + (int)floor(local_mem_coord.x);
			// squared difference
			kernel_pix_idx = (kernel_pivot_idx.y + dy) * kernel_size.x + (kernel_pivot_idx.x + dx);			
			diff = image_local_buffer[local_buffer_idx] - kernel_tex[kernel_pix_idx];
			sqdiff += dot(diff, diff);			
		}
	}

	// write result
	int2 out_coord = (int2)(gid_x, gid_y);
	write_imagef(response_tex, out_coord, (float4)(sqdiff, 0.0f, 0.0f, 0.0f) + read_imagef(prev_response_tex, kernel_sampler, out_coord));
}