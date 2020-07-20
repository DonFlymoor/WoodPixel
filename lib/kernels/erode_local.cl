const sampler_t mask_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void erode_local(
	__read_only image2d_t input_tex,
	__write_only image2d_t response_tex,
	__local float* local_buffer,
	int2 input_size,
	int2 output_size,
	int2 kernel_size,
	int2 kernel_anchor,
	int4 kernel_overlaps,
	float2 rotation_sincos)
{
	const int gid_x = get_global_id(0);
	const int gid_y = get_global_id(1);
	const int lid_x = get_local_id(0);
	const int lid_y = get_local_id(1);
	const int ls_x = get_local_size(0);
	const int ls_y = get_local_size(1);
	const int lid = lid_y * ls_x + lid_x;

	// "center" index of the kernel. This is also the offset into the input image.
	const int2 kernel_start_idx = -kernel_anchor;
	const int2 kernel_end_idx = kernel_size - kernel_anchor - 1;
	
	const float2 input_pivot = (float2)(
		(float)(gid_x) + 0.5f,
		(float)(gid_y) + 0.5f
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
	local_buffer[(ht + lid_y) * local_buffer_width + (hl + lid_x)] = read_imagef(input_tex, mask_sampler, input_pivot).x;
	// load overlap texels
	// vertical edges
	if(lid_x < hl)
		local_buffer[(ht + lid_y) * local_buffer_width + (lid_x)] = read_imagef(input_tex, mask_sampler, input_pivot - (float2)((float)hl, 0.0f)).x;
	if(lid_x >= (ls_x - hr))
		local_buffer[(ht + lid_y) * local_buffer_width + (hl + lid_x + hr)] = read_imagef(input_tex, mask_sampler, input_pivot + (float2)((float)hr, 0.0f)).x;
	// horizontal edges 
	if(lid_y < ht)
		local_buffer[(lid_y) * local_buffer_width + (hl + lid_x)] = read_imagef(input_tex, mask_sampler, input_pivot - (float2)(0.0f, (float)ht)).x;
	if(lid_y >= (ls_y - hb))
		local_buffer[(ht + lid_y + hb) * local_buffer_width + (hl + lid_x)] = read_imagef(input_tex, mask_sampler, input_pivot + (float2)(0.0f, (float)hb)).x;
	// // four corners
	if((lid_x < hl) && (lid_y < ht)) // upper left corner
		local_buffer[(lid_y) * local_buffer_width + (lid_x)] = read_imagef(input_tex, mask_sampler, input_pivot - (float2)((float)hl, (float)ht)).x;
	if((lid_x >= (ls_x - hr)) && (lid_y < ht)) // upper right corner
		local_buffer[(lid_y) * local_buffer_width + (hl + lid_x + hr)] = read_imagef(input_tex, mask_sampler, input_pivot + (float2)((float)hr, (float)-ht)).x;
	if((lid_x < hl) && (lid_y >= (ls_y - hb))) // lower left corner
		local_buffer[(ht + lid_y + hb) * local_buffer_width + (lid_x)] = read_imagef(input_tex, mask_sampler, input_pivot + (float2)((float)-hl, (float)hb)).x;
	if((lid_x >= (ls_x - hr)) && (lid_y >= (ls_y - hb))) // lower right corner
		local_buffer[(ht + lid_y + hb) * local_buffer_width + (hl + lid_x + hr)] = read_imagef(input_tex, mask_sampler, input_pivot + (float2)((float)hr, (float)hb)).x;

	// synchronize all local buffer writes
	barrier(CLK_LOCAL_MEM_FENCE);
	if(gid_x < output_size.x && gid_y < output_size.y)
	{
		float minval = 1.0f;
		float2 cdelta = (float2)(0.0f);
		// iterate over kernel area
		int local_buffer_idx;
		float2 local_mem_coord;
		for(int dy = kernel_start_idx.y; dy != kernel_end_idx.y; ++dy)
		{
			for(int dx = kernel_start_idx.x; dx != kernel_end_idx.x; ++dx)
			{
				cdelta = (float2)((float)dx, (float)dy);
				
				// calculate image coord (applies rotation around current texel!)
				local_mem_coord.x = rotation_sincos.y * cdelta.x - rotation_sincos.x * cdelta.y + local_mem_pivot.x;
				local_mem_coord.y = rotation_sincos.x * cdelta.x + rotation_sincos.y * cdelta.y + local_mem_pivot.y;
				local_buffer_idx = (int)floor(local_mem_coord.y) * local_buffer_width + (int)floor(local_mem_coord.x);
				
				float image_val = step(0.5f, local_buffer[local_buffer_idx]);			
				minval = min(minval, image_val);				
			}
		}
		// write result
		write_imagef(response_tex, (int2)(gid_x, gid_y), (float4)(minval, 0.0f, 0.0f, 0.0f));
	}
}