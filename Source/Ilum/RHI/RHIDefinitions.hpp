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

enum class RHIShaderStage
{
	// Rasterization
	Vertex,
	TessellationControl,
	TessellationEvaluation,
	Geometry,
	Fragment,

	// Compute
	Compute,

	// Ray Tracing
	RayGen,
	AnyHit,
	ClosestHit,
	Miss,
	Intersection,
	Callable,

	// Mesh Shading
	Mesh,
	Task
};
DEFINE_ENUMCLASS_OPERATION(RHIShaderStage)

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

	R16_UINT,
	R16_SINT,
	R16_FLOAT,

	R8G8B8A8_UNORM,
	B8G8R8A8_UNORM,

	R32_UINT,
	R32_SINT,
	R32_FLOAT,

	D32_FLOAT,
	D24_UNORM_S8_UINT,

	R16G16_UINT,
	R16G16_SINT,
	R16G16_FLOAT,

	R16G16B16A16_UINT,
	R16G16B16A16_SINT,
	R16G16B16A16_FLOAT,

	R32G32_UINT,
	R32G32_SINT,
	R32G32_FLOAT,

	R32G32B32_UINT,
	R32G32B32_SINT,
	R32G32B32_FLOAT,

	R32G32B32A32_UINT,
	R32G32B32A32_SINT,
	R32G32B32A32_FLOAT
};

inline bool IsDepthFormat(RHIFormat format)
{
	return format == RHIFormat::D32_FLOAT ||
	       format == RHIFormat::D24_UNORM_S8_UINT;
}

inline bool IsStencilFormat(RHIFormat format)
{
	return format == RHIFormat::D24_UNORM_S8_UINT;
}

enum class RHIVertexSemantics
{
	Binormal,
	Blend_Indices,
	Blend_Weights,
	Color,
	Normal,
	Position,
	Tangent,
	Texcoord
};

enum class RHIMemoryUsage
{
	GPU_Only,
	CPU_TO_GPU,
	GPU_TO_CPU
};

// Texture
enum class RHITextureDimension : uint64_t
{
	Texture1D        = 1,
	Texture2D        = 1 << 1,
	Texture3D        = 1 << 2,
	TextureCube      = 1 << 3,
	Texture1DArray   = 1 << 4,
	Texture2DArray   = 1 << 5,
	TextureCubeArray = 1 << 6,
};

enum class RHITextureUsage
{
	Undefined       = 0,
	Transfer        = 1,
	ShaderResource  = 1 << 1,
	UnorderedAccess = 1 << 2,
	RenderTarget    = 1 << 3
};
DEFINE_ENUMCLASS_OPERATION(RHITextureUsage)

enum class RHIResourceState
{
	Undefined,
	VertexBuffer,
	ConstantBuffer,
	IndexBuffer,
	IndirectBuffer,
	TransferSource,
	TransferDest,
	ShaderResource,
	UnorderedAccess,
	RenderTarget,
	DepthWrite,
	DepthRead,
	AccelerationStructure,
	Present
};

// Buffer
enum class RHIBufferUsage
{
	Undefined             = 0,
	Vertex                = 1,
	Index                 = 1 << 1,
	Indirect              = 1 << 2,
	Transfer              = 1 << 3,
	AccelerationStructure = 1 << 4,
	ShaderResource        = 1 << 5,
	UnorderedAccess       = 1 << 6,
	ConstantBuffer        = 1 << 7
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

enum class RHISamplerBorderColor
{
	Float_Transparent_Black,
	Int_Transparent_Black,
	Float_Opaque_Black,
	Int_Opaque_Black,
	Float_Opaque_White,
	Int_Opaque_White
};

// Pipeline State
enum class RHICompareOp
{
	Never,
	Less,
	Equal,
	Less_Or_Equal,
	Greater,
	Not_Equal,
	Greater_Or_Equal,
	Always
};

enum class RHILogicOp
{
	Clear,
	And,
	And_Reverse,
	Copy,
	And_Inverted,
	No_Op,
	XOR,
	Or,
	Nor,
	Equivalent,
	Invert,
	Or_Reverse,
	Copy_Inverted,
	Or_Inverted,
	Nand,
	Set
};

enum class RHIBlendFactor
{
	Zero,
	One,
	Src_Color,
	One_Minus_Src_Color,
	Dst_Color,
	One_Minus_Dst_Color,
	Src_Alpha,
	One_Minus_Src_Alpha,
	Dst_Alpha,
	One_Minus_Dst_Alpha,
	Constant_Color,
	One_Minus_Constant_Color,
	Constant_Alpha,
	One_Minus_Constant_Alpha,
	Src_Alpha_Saturate,
	Src1_Color,
	One_Minus_Src1_Color,
	Src1_Alpha,
	One_Minus_Src1_Alpha
};

enum class RHIBlendOp
{
	Add,
	Subtract,
	Reverse_Subtract,
	Min,
	Max
};

enum class RHICullMode
{
	None,
	Front,
	Back
};

enum class RHIFrontFace
{
	Counter_Clockwise,
	Clockwise
};

enum class RHIPolygonMode
{
	Wireframe,
	Solid
};

enum class RHIPrimitiveTopology
{
	Point,
	Line,
	Triangle,
	Patch
};

enum class RHIVertexInputRate
{
	Vertex,
	Instance
};

enum class RHILoadAction
{
	DontCare,
	Load,
	Clear
};

enum class RHIStoreAction
{
	DontCare,
	Store
};
}        // namespace Ilum
