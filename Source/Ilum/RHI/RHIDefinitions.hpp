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
	Unknown,
	Vulkan,
	DX12
};

enum class RHIFeature
{
	RayTracing,
	MeshShading,
	BufferDeviceAddress,
	Bindless
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
	D24_UNORM_S8_UINT
};

inline bool IsDepthFormat(RHIFormat format)
{
	return format == RHIFormat::D32_SFLOAT ||
	       format == RHIFormat::D24_UNORM_S8_UINT;
}

enum class RHIMemoryUsage
{
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
	ShaderResource,
	UnorderedAccess,
	RenderTarget
};
DEFINE_ENUMCLASS_OPERATION(RHITextureUsage)

enum class RHITextureState
{
	Undefined,
	TransferSource,
	TransferDest,
	ShaderResource,
	UnorderAccess,
	RenderTarget,
	DepthWrite,
	DepthRead,
	Present
};

// Buffer
enum class RHIBufferUsage
{
	Undefined,
	Vertex,
	Index,
	Indirect,
	Transfer,
	AccelerationStructure,
	ShaderResource,
	UnorderedAccess,
	ConstantBuffer
};
DEFINE_ENUMCLASS_OPERATION(RHIBufferUsage)

enum class RHIBufferState
{
	Undefined,
	Vertex,
	Index,
	Indirect,
	TransferSource,
	TransferDest,
	AccelerationStructure,
	ShaderResource,
	UnorderedAccess,
	ConstantBuffer
};

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

enum class RHISamplerBorderColor
{
	Float_Transparent_Black,
	Int_Transparent_Black,
	Float_Opaque_Black,
	Int_Opaque_Black,
	Float_Opaque_White,
	Int_Opaque_White
};
}        // namespace Ilum
