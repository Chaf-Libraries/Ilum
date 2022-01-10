#ifndef COMMON_GLSL
#define COMMON_GLSL

#define PI 3.141592653589793

float RadicalInverse_VdC(uint bits) 
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

vec2 Hammersley(uint i, uint N)
{
    return vec2(float(i)/float(N), RadicalInverse_VdC(i));
}

vec2 compute_motion_vector(vec4 prev_pos, vec4 current_pos)
{
    // Clip space -> NDC
    vec2 current = current_pos.xy / current_pos.w;
    vec2 prev = prev_pos.xy / prev_pos.w;

    current = current * 0.5 + 0.5;
    prev = prev * 0.5 + 0.5;

    current.y = 1 - current.y;
    prev.y = 1 - prev.y;

    return current - prev;
}

#endif