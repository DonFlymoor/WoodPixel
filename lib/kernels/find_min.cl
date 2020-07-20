const sampler_t mask_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void find_min_masked(
	__read_only image2d_t input_tex,
	__read_only image2d_t texture_mask,
	__global float4* response,
    __local float4* local_buffer,
	int2 input_size,
    int2 texture_mask_offset)
{
	const int2 gid = (int2)(get_global_id(0), get_global_id(1));
    const float2 imcoord = (float2)((float)gid.x, (float)gid.y);
    const int2 lid = (int2)(get_local_id(0), get_local_id(1));
    const int2 block_size = (int2)(get_local_size(0), get_local_size(1));
    const int local_index = lid.y * block_size.x + lid.x;    
    const int group_size = block_size.x * block_size.y;
    const int group_index = get_group_id(1) * get_num_groups(0) + get_group_id(0);

    // first collect data in local memory (cost value, mask value, minpos_x, minpos_y)
    float costval = read_imagef(input_tex, mask_sampler, gid).x;
    float maskval = read_imagef(texture_mask, mask_sampler, gid + texture_mask_offset).x;
    // valid texel?  && (maskval > 0.5f)
    local_buffer[local_index] = (gid.x < input_size.x && gid.y < input_size.y) ? (float4)(costval, maskval, imcoord.x, imcoord.y) : (float4)(FLT_MAX, 0.0f, 0.0f, 0.0f);
    // parallel reduction
    for(int stride = group_size / 2; stride > 0; stride /= 2)
    {
        // synchronize local_buffer_writes
        barrier(CLK_LOCAL_MEM_FENCE);
        if(local_index < stride)
        {
            float4 lhs = local_buffer[local_index];
            float4 rhs = local_buffer[local_index + stride];
            // avoid diverging branches
            float selector = mix(step(lhs.y, rhs.y), step(rhs.x, lhs.x), lhs.y * rhs.y) * min(lhs.y + rhs.y, 1.0f);
            local_buffer[local_index] = mix(lhs, rhs, selector);
        }
    }
    // write out result
    if(local_index == 0)
        response[group_index] = local_buffer[0];
}

__kernel void find_min(
	__read_only image2d_t input_tex,
	__global float4* response,
    __local float4* local_buffer,
	int2 input_size)
{
	const int2 gid = (int2)(get_global_id(0), get_global_id(1));
    const float2 imcoord = (float2)((float)gid.x, (float)gid.y);
    const int2 lid = (int2)(get_local_id(0), get_local_id(1));
    const int2 block_size = (int2)(get_local_size(0), get_local_size(1));
    const int local_index = lid.y * block_size.x + lid.x;    
    const int group_size = block_size.x * block_size.y;
    const int group_index = get_group_id(1) * get_num_groups(0) + get_group_id(0);

    // first collect data in local memory (cost value, mask value, minpos_x, minpos_y)
    float cost_val = read_imagef(input_tex, mask_sampler, gid).x;
    local_buffer[local_index] = (gid.x < input_size.x && gid.y < input_size.y) ? (float4)(cost_val, 1.0f, imcoord.x, imcoord.y) : (float4)(FLT_MAX, 0.0f, 0.0f, 0.0f);
    // parallel reduction
    for(int stride = group_size / 2; stride > 0; stride /= 2)
    {
        // synchronize local_buffer_writes
        barrier(CLK_LOCAL_MEM_FENCE);
        if(local_index < stride)
        {
            float4 lhs = local_buffer[local_index];
            float4 rhs = local_buffer[local_index + stride];
            // avoid diverging branches
            float selector = step(rhs.x, lhs.x);
            local_buffer[local_index] = mix(lhs, rhs, selector);
        }
    }
    // write out result
    if(local_index == 0)
        response[group_index] = local_buffer[0];
}