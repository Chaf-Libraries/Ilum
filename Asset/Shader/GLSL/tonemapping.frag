#version 450

layout(binding = 0) uniform sampler2D inColor;

layout (location = 0) in vec2 inUV;

layout (location = 0) out vec4 outColor;

layout(push_constant) uniform PushBlock{
    float exposure;
    float gamma;
}push_data;

// From http://filmicgames.com/archives/75
vec3 uncharted2Tonemap(vec3 x)
{
	float A = 0.15;
	float B = 0.50;
	float C = 0.10;
	float D = 0.20;
	float E = 0.02;
	float F = 0.30;
	return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;
}

void main()
{
    vec3 color = texture(inColor, inUV).rgb;
    color = uncharted2Tonemap(color * push_data.exposure);
	color = color * (1.0f / uncharted2Tonemap(vec3(push_data.gamma)));	
	// Gamma correction
	color = pow(color, vec3(1.0f / 2.2));

    outColor = vec4(color, 1.0);
}