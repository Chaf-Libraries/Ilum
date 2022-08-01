#pragma once

namespace Ilum
{
#define DEFINE_ENUMCLASS_OPERATION(EnumClass)                 \
	inline EnumClass operator|(EnumClass lhs, EnumClass rhs)  \
	{                                                         \
		return (EnumClass) ((uint64_t) lhs | (uint64_t) rhs); \
	}                                                         \
	inline bool operator&(EnumClass lhs, EnumClass rhs)       \
	{                                                         \
		return (bool) ((uint64_t) lhs & (uint64_t) rhs);      \
	}

enum class RHIBackend
{
	Vulkan,
	// TODO:
	// DX12
	// OpenGL
};

enum class RHIQueueFamily
{
	Graphics,
	Compute,
	Transfer
};

enum class RHIFormat
{
	Undefined,

	R8G8B8A8_UNORM,

	R16G16B16A16_SFLOAT,
	R32G32B32A32_SFLOAT,

	D32_SFLOAT,
	D32_SFLOAT_S8_UINT
};

inline bool IsDepthFormat(RHIFormat format)
{
	return format == RHIFormat::D32_SFLOAT ||
	       format == RHIFormat::D32_SFLOAT_S8_UINT;
}

enum class RHIMemoryUsage
{
	CPU_Only,
	GPU_Only,
	CPU_TO_GPU,
	GPU_TO_CPU
};

// Texture
enum class RHITextureDimension
{
	Texture1D,
	Texture2D,
	Texture3D,
	TextureCube,
	Texture1DArray,
	Texture2DArray,
	TextureCubeArray
};

enum class RHITextureUsage
{
	Transfer,
	SRV,
	UAV,
	RenderTarget
};
DEFINE_ENUMCLASS_OPERATION(RHITextureUsage)

// Buffer
enum class RHIBufferUsage
{
	Uniform,
	Vertex,
	Index,
	Indirect,
	Transfer,
	AccelerationStructure,
	SRV,
	UAV
};
DEFINE_ENUMCLASS_OPERATION(RHIBufferUsage)

// Sampler
enum class RHIFilter
{
	Nearest,
	Linear
};

enum class RHIAddressMode
{
	Repeat,
	Mirrored_Repeat,
	Clamp_To_Edge,
	Clamp_To_Border,
	Mirror_Clamp_To_Edge
};

enum class RHIMipmapMode
{
	Nearest,
	Linear
};
}        // namespace Ilum
