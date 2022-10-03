#pragma once

#include <Core/Macro.hpp>

namespace Ilum
{
#define DEFINE_ENUMCLASS_OPERATION(EnumClass)                   \
	inline EnumClass operator|(EnumClass lhs, EnumClass rhs)    \
	{                                                           \
		return (EnumClass) ((uint64_t) lhs | (uint64_t) rhs);   \
	}                                                           \
	inline bool operator&(EnumClass lhs, EnumClass rhs)         \
	{                                                           \
		return (bool) ((uint64_t) lhs & (uint64_t) rhs);        \
	}                                                           \
	inline EnumClass &operator|=(EnumClass &lhs, EnumClass rhs) \
	{                                                           \
		return lhs = lhs | rhs;                                 \
	}

ENUM(RHIBackend, Enable){
    Unknown,
    Vulkan,
    DX12,
    OpenGL,
    CUDA};

ENUM(RHIShaderStage, Enable){
    Vertex                 = 1,
    TessellationControl    = 1 << 1,
    TessellationEvaluation = 1 << 2,
    Geometry               = 1 << 3,
    Fragment               = 1 << 4,
    Compute                = 1 << 5,
    RayGen                 = 1 << 6,
    AnyHit                 = 1 << 7,
    ClosestHit             = 1 << 8,
    Miss                   = 1 << 9,
    Intersection           = 1 << 10,
    Callable               = 1 << 11,
    Mesh                   = 1 << 12,
    Task                   = 1 << 13};

DEFINE_ENUMCLASS_OPERATION(RHIShaderStage);

ENUM(RHIFeature, Enable){
    RayTracing,
    MeshShading,
    BufferDeviceAddress,
    Bindless};

ENUM(RHIQueueFamily, Enable){
	Graphics,
	Compute,
	Transfer
};

ENUM(RHIFormat, Enable){
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
	R10G10B10A2_UNORM,
	R10G10B10A2_UINT,
	R11G11B10_FLOAT,
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

ENUM(RHIVertexSemantics, Enable){
	Binormal,
	Blend_Indices,
	Blend_Weights,
	Color,
	Normal,
	Position,
	Tangent,
	Texcoord
};

ENUM(RHIMemoryUsage, Enable){
	GPU_Only,
	CPU_TO_GPU,
	GPU_TO_CPU
};

// Texture
ENUM(RHITextureDimension, Enable) : uint64_t
{
	Texture1D        = 1,
	Texture2D        = 1 << 1,
	Texture3D        = 1 << 2,
	TextureCube      = 1 << 3,
	Texture1DArray   = 1 << 4,
	Texture2DArray   = 1 << 5,
	TextureCubeArray = 1 << 6,
};

inline RHITextureDimension GetTextureDimension(uint32_t width, uint32_t height, uint32_t depth, uint32_t layers)
{
	if (layers == 1)
	{
		if (depth == 1)
		{
			if (height == 1)
			{
				return RHITextureDimension::Texture1D;
			}
			else
			{
				return RHITextureDimension::Texture2D;
			}
		}
		else
		{
			return RHITextureDimension::Texture3D;
		}
	}

	if (layers == 6)
	{
		return RHITextureDimension::TextureCube;
	}

	if (layers % 6 == 0)
	{
		return RHITextureDimension::TextureCubeArray;
	}

	if (height == 1)
	{
		return RHITextureDimension::Texture1DArray;
	}
	else
	{
		return RHITextureDimension::Texture2DArray;
	}
}

// Buffer
ENUM(RHIBufferUsage, Enable){
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
DEFINE_ENUMCLASS_OPERATION(RHIBufferUsage);

ENUM(RHITextureUsage, Enable){
	Undefined       = 0,
	Transfer        = 1,
	ShaderResource  = 1 << 1,
	UnorderedAccess = 1 << 2,
	RenderTarget    = 1 << 3
};

DEFINE_ENUMCLASS_OPERATION(RHITextureUsage);

ENUM(RHIResourceState, Enable){
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

inline RHITextureUsage ResourceStateToTextureUsage(RHIResourceState state)
{
	switch (state)
	{
		case RHIResourceState::TransferSource:
		case RHIResourceState::TransferDest:
			return RHITextureUsage::Transfer;
		case RHIResourceState::ShaderResource:
			return RHITextureUsage::ShaderResource;
		case RHIResourceState::UnorderedAccess:
			return RHITextureUsage::UnorderedAccess;
		case RHIResourceState::RenderTarget:
		case RHIResourceState::DepthWrite:
			return RHITextureUsage::RenderTarget;
		case RHIResourceState::DepthRead:
			return RHITextureUsage::ShaderResource;
		default:
			break;
	}

	return RHITextureUsage::Undefined;
}

inline RHIBufferUsage ResourceStateToBufferUsage(RHIResourceState state)
{
	switch (state)
	{
		case RHIResourceState::VertexBuffer:
			return RHIBufferUsage::Vertex;
		case RHIResourceState::ConstantBuffer:
			return RHIBufferUsage::ConstantBuffer;
		case RHIResourceState::IndexBuffer:
			return RHIBufferUsage::Index;
		case RHIResourceState::IndirectBuffer:
		case RHIResourceState::TransferSource:
		case RHIResourceState::TransferDest:
			return RHIBufferUsage::Transfer;
		case RHIResourceState::ShaderResource:
			return RHIBufferUsage::ShaderResource;
		case RHIResourceState::UnorderedAccess:
			return RHIBufferUsage::UnorderedAccess;
		case RHIResourceState::AccelerationStructure:
			return RHIBufferUsage::AccelerationStructure;
		default:
			break;
	}
	return RHIBufferUsage::Undefined;
}

// Sampler
ENUM(RHIFilter, Enable){
	Nearest,
	Linear
};

ENUM(RHIAddressMode, Enable){
	Repeat,
	Mirrored_Repeat,
	Clamp_To_Edge,
	Clamp_To_Border,
	Mirror_Clamp_To_Edge
};

ENUM(RHIMipmapMode, Enable){
	Nearest,
	Linear
};

ENUM(RHISamplerBorderColor, Enable){
	Float_Transparent_Black,
	Int_Transparent_Black,
	Float_Opaque_Black,
	Int_Opaque_Black,
	Float_Opaque_White,
	Int_Opaque_White
};

// Pipeline State
ENUM(RHICompareOp, Enable){
	Never,
	Less,
	Equal,
	Less_Or_Equal,
	Greater,
	Not_Equal,
	Greater_Or_Equal,
	Always
};

ENUM(RHILogicOp, Enable){
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

ENUM(RHIBlendFactor, Enable){
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

ENUM(RHIBlendOp, Enable){
	Add,
	Subtract,
	Reverse_Subtract,
	Min,
	Max
};

ENUM(RHICullMode, Enable){
	None,
	Front,
	Back
};

ENUM(RHIFrontFace, Enable){
	Counter_Clockwise,
	Clockwise
};

ENUM(RHIPolygonMode, Enable)
{
	Wireframe,
	Solid
};

ENUM(RHIPrimitiveTopology, Enable){
	Point,
	Line,
	Triangle,
	Patch
};

ENUM(RHIVertexInputRate, Enable){
	Vertex,
	Instance
};

ENUM(RHILoadAction, Enable){
	DontCare,
	Load,
	Clear
};

ENUM(RHIStoreAction, Enable){
	DontCare,
	Store
};
}        // namespace Ilum