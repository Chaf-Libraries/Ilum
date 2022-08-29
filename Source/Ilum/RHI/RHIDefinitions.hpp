#pragma once

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

REFLECTION_ENUM RHIBackend{
    Unknown,
    Vulkan,
    DX12};

REFLECTION_ENUM RHIShaderStage{
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
DEFINE_ENUMCLASS_OPERATION(RHIShaderStage)

REFLECTION_ENUM RHIFeature{
    RayTracing,
    MeshShading,
    BufferDeviceAddress,
    Bindless};

REFLECTION_ENUM RHIQueueFamily{
    Graphics,
    Compute,
    Transfer};

REFLECTION_ENUM RHIFormat{
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
    R32G32B32A32_FLOAT};

inline bool IsDepthFormat(RHIFormat format)
{
	return format == RHIFormat::D32_FLOAT ||
	       format == RHIFormat::D24_UNORM_S8_UINT;
}

inline bool IsStencilFormat(RHIFormat format)
{
	return format == RHIFormat::D24_UNORM_S8_UINT;
}

REFLECTION_ENUM RHIVertexSemantics{
    Binormal,
    Blend_Indices,
    Blend_Weights,
    Color,
    Normal,
    Position,
    Tangent,
    Texcoord};

REFLECTION_ENUM RHIMemoryUsage{
    GPU_Only,
    CPU_TO_GPU,
    GPU_TO_CPU};

// Texture
REFLECTION_ENUM RHITextureDimension : uint64_t{
                                          Texture1D        = 1,
                                          Texture2D        = 1 << 1,
                                          Texture3D        = 1 << 2,
                                          TextureCube      = 1 << 3,
                                          Texture1DArray   = 1 << 4,
                                          Texture2DArray   = 1 << 5,
                                          TextureCubeArray = 1 << 6,
                                      };

REFLECTION_ENUM RHITextureUsage{
    Undefined       = 0,
    Transfer        = 1,
    ShaderResource  = 1 << 1,
    UnorderedAccess = 1 << 2,
    RenderTarget    = 1 << 3};
DEFINE_ENUMCLASS_OPERATION(RHITextureUsage)

REFLECTION_ENUM RHIResourceState{
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
    Present};

// Buffer
REFLECTION_ENUM RHIBufferUsage{
    Undefined             = 0,
    Vertex                = 1,
    Index                 = 1 << 1,
    Indirect              = 1 << 2,
    Transfer              = 1 << 3,
    AccelerationStructure = 1 << 4,
    ShaderResource        = 1 << 5,
    UnorderedAccess       = 1 << 6,
    ConstantBuffer        = 1 << 7};
DEFINE_ENUMCLASS_OPERATION(RHIBufferUsage)

// Sampler
REFLECTION_ENUM RHIFilter{
    Nearest,
    Linear};

REFLECTION_ENUM RHIAddressMode{
    Repeat,
    Mirrored_Repeat,
    Clamp_To_Edge,
    Clamp_To_Border,
    Mirror_Clamp_To_Edge};

REFLECTION_ENUM RHIMipmapMode{
    Nearest,
    Linear};

REFLECTION_ENUM RHISamplerBorderColor{
    Float_Transparent_Black,
    Int_Transparent_Black,
    Float_Opaque_Black,
    Int_Opaque_Black,
    Float_Opaque_White,
    Int_Opaque_White};

// Pipeline State
REFLECTION_ENUM RHICompareOp{
    Never,
    Less,
    Equal,
    Less_Or_Equal,
    Greater,
    Not_Equal,
    Greater_Or_Equal,
    Always};

REFLECTION_ENUM RHILogicOp{
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
    Set};

REFLECTION_ENUM RHIBlendFactor{
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
    One_Minus_Src1_Alpha};

REFLECTION_ENUM RHIBlendOp{
    Add,
    Subtract,
    Reverse_Subtract,
    Min,
    Max};

REFLECTION_ENUM RHICullMode{
    None,
    Front,
    Back};

REFLECTION_ENUM RHIFrontFace{
    Counter_Clockwise,
    Clockwise};

REFLECTION_ENUM RHIPolygonMode{
    Wireframe,
    Solid};

REFLECTION_ENUM RHIPrimitiveTopology{
    Point,
    Line,
    Triangle,
    Patch};

REFLECTION_ENUM RHIVertexInputRate{
    Vertex,
    Instance};

REFLECTION_ENUM RHILoadAction{
    DontCare,
    Load,
    Clear};

REFLECTION_ENUM RHIStoreAction{
    DontCare,
    Store};
}        // namespace Ilum
