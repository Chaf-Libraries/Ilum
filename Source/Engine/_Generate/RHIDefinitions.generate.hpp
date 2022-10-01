#include "E:/Workspace/Ilum/Source/Runtime/RHI/Public/RHI/RHIDefinitions.hpp"
#include <rttr/registration.h>

namespace Ilum_7328119563694619655
{
RTTR_REGISTRATION
{
    rttr::registration::enumeration<Ilum::RHIBackend>("RHIBackend")
    (
        rttr::value("Unknown", Ilum::RHIBackend::Unknown),
        rttr::value("Vulkan", Ilum::RHIBackend::Vulkan),
        rttr::value("DX12", Ilum::RHIBackend::DX12),
        rttr::value("OpenGL", Ilum::RHIBackend::OpenGL),
        rttr::value("CUDA", Ilum::RHIBackend::CUDA)
    );
    rttr::registration::enumeration<Ilum::RHIShaderStage>("RHIShaderStage")
    (
        rttr::value("Vertex", Ilum::RHIShaderStage::Vertex),
        rttr::value("TessellationControl", Ilum::RHIShaderStage::TessellationControl),
        rttr::value("TessellationEvaluation", Ilum::RHIShaderStage::TessellationEvaluation),
        rttr::value("Geometry", Ilum::RHIShaderStage::Geometry),
        rttr::value("Fragment", Ilum::RHIShaderStage::Fragment),
        rttr::value("Compute", Ilum::RHIShaderStage::Compute),
        rttr::value("RayGen", Ilum::RHIShaderStage::RayGen),
        rttr::value("AnyHit", Ilum::RHIShaderStage::AnyHit),
        rttr::value("ClosestHit", Ilum::RHIShaderStage::ClosestHit),
        rttr::value("Miss", Ilum::RHIShaderStage::Miss),
        rttr::value("Intersection", Ilum::RHIShaderStage::Intersection),
        rttr::value("Callable", Ilum::RHIShaderStage::Callable),
        rttr::value("Mesh", Ilum::RHIShaderStage::Mesh),
        rttr::value("Task", Ilum::RHIShaderStage::Task)
    );
    rttr::registration::enumeration<Ilum::RHIFeature>("RHIFeature")
    (
        rttr::value("RayTracing", Ilum::RHIFeature::RayTracing),
        rttr::value("MeshShading", Ilum::RHIFeature::MeshShading),
        rttr::value("BufferDeviceAddress", Ilum::RHIFeature::BufferDeviceAddress),
        rttr::value("Bindless", Ilum::RHIFeature::Bindless)
    );
    rttr::registration::enumeration<Ilum::RHIQueueFamily>("RHIQueueFamily")
    (
        rttr::value("Graphics", Ilum::RHIQueueFamily::Graphics),
        rttr::value("Compute", Ilum::RHIQueueFamily::Compute),
        rttr::value("Transfer", Ilum::RHIQueueFamily::Transfer)
    );
    rttr::registration::enumeration<Ilum::RHIFormat>("RHIFormat")
    (
        rttr::value("Undefined", Ilum::RHIFormat::Undefined),
        rttr::value("R16_UINT", Ilum::RHIFormat::R16_UINT),
        rttr::value("R16_SINT", Ilum::RHIFormat::R16_SINT),
        rttr::value("R16_FLOAT", Ilum::RHIFormat::R16_FLOAT),
        rttr::value("R8G8B8A8_UNORM", Ilum::RHIFormat::R8G8B8A8_UNORM),
        rttr::value("B8G8R8A8_UNORM", Ilum::RHIFormat::B8G8R8A8_UNORM),
        rttr::value("R32_UINT", Ilum::RHIFormat::R32_UINT),
        rttr::value("R32_SINT", Ilum::RHIFormat::R32_SINT),
        rttr::value("R32_FLOAT", Ilum::RHIFormat::R32_FLOAT),
        rttr::value("D32_FLOAT", Ilum::RHIFormat::D32_FLOAT),
        rttr::value("D24_UNORM_S8_UINT", Ilum::RHIFormat::D24_UNORM_S8_UINT),
        rttr::value("R16G16_UINT", Ilum::RHIFormat::R16G16_UINT),
        rttr::value("R16G16_SINT", Ilum::RHIFormat::R16G16_SINT),
        rttr::value("R16G16_FLOAT", Ilum::RHIFormat::R16G16_FLOAT),
        rttr::value("R10G10B10A2_UNORM", Ilum::RHIFormat::R10G10B10A2_UNORM),
        rttr::value("R10G10B10A2_UINT", Ilum::RHIFormat::R10G10B10A2_UINT),
        rttr::value("R11G11B10_FLOAT", Ilum::RHIFormat::R11G11B10_FLOAT),
        rttr::value("R16G16B16A16_UINT", Ilum::RHIFormat::R16G16B16A16_UINT),
        rttr::value("R16G16B16A16_SINT", Ilum::RHIFormat::R16G16B16A16_SINT),
        rttr::value("R16G16B16A16_FLOAT", Ilum::RHIFormat::R16G16B16A16_FLOAT),
        rttr::value("R32G32_UINT", Ilum::RHIFormat::R32G32_UINT),
        rttr::value("R32G32_SINT", Ilum::RHIFormat::R32G32_SINT),
        rttr::value("R32G32_FLOAT", Ilum::RHIFormat::R32G32_FLOAT),
        rttr::value("R32G32B32_UINT", Ilum::RHIFormat::R32G32B32_UINT),
        rttr::value("R32G32B32_SINT", Ilum::RHIFormat::R32G32B32_SINT),
        rttr::value("R32G32B32_FLOAT", Ilum::RHIFormat::R32G32B32_FLOAT),
        rttr::value("R32G32B32A32_UINT", Ilum::RHIFormat::R32G32B32A32_UINT),
        rttr::value("R32G32B32A32_SINT", Ilum::RHIFormat::R32G32B32A32_SINT),
        rttr::value("R32G32B32A32_FLOAT", Ilum::RHIFormat::R32G32B32A32_FLOAT)
    );
    rttr::registration::enumeration<Ilum::RHIVertexSemantics>("RHIVertexSemantics")
    (
        rttr::value("Binormal", Ilum::RHIVertexSemantics::Binormal),
        rttr::value("Blend_Indices", Ilum::RHIVertexSemantics::Blend_Indices),
        rttr::value("Blend_Weights", Ilum::RHIVertexSemantics::Blend_Weights),
        rttr::value("Color", Ilum::RHIVertexSemantics::Color),
        rttr::value("Normal", Ilum::RHIVertexSemantics::Normal),
        rttr::value("Position", Ilum::RHIVertexSemantics::Position),
        rttr::value("Tangent", Ilum::RHIVertexSemantics::Tangent),
        rttr::value("Texcoord", Ilum::RHIVertexSemantics::Texcoord)
    );
    rttr::registration::enumeration<Ilum::RHIMemoryUsage>("RHIMemoryUsage")
    (
        rttr::value("GPU_Only", Ilum::RHIMemoryUsage::GPU_Only),
        rttr::value("CPU_TO_GPU", Ilum::RHIMemoryUsage::CPU_TO_GPU),
        rttr::value("GPU_TO_CPU", Ilum::RHIMemoryUsage::GPU_TO_CPU)
    );
    rttr::registration::enumeration<Ilum::RHITextureDimension>("RHITextureDimension")
    (
        rttr::value("Texture1D", Ilum::RHITextureDimension::Texture1D),
        rttr::value("Texture2D", Ilum::RHITextureDimension::Texture2D),
        rttr::value("Texture3D", Ilum::RHITextureDimension::Texture3D),
        rttr::value("TextureCube", Ilum::RHITextureDimension::TextureCube),
        rttr::value("Texture1DArray", Ilum::RHITextureDimension::Texture1DArray),
        rttr::value("Texture2DArray", Ilum::RHITextureDimension::Texture2DArray),
        rttr::value("TextureCubeArray", Ilum::RHITextureDimension::TextureCubeArray)
    );
    rttr::registration::enumeration<Ilum::RHIBufferUsage>("RHIBufferUsage")
    (
        rttr::value("Undefined", Ilum::RHIBufferUsage::Undefined),
        rttr::value("Vertex", Ilum::RHIBufferUsage::Vertex),
        rttr::value("Index", Ilum::RHIBufferUsage::Index),
        rttr::value("Indirect", Ilum::RHIBufferUsage::Indirect),
        rttr::value("Transfer", Ilum::RHIBufferUsage::Transfer),
        rttr::value("AccelerationStructure", Ilum::RHIBufferUsage::AccelerationStructure),
        rttr::value("ShaderResource", Ilum::RHIBufferUsage::ShaderResource),
        rttr::value("UnorderedAccess", Ilum::RHIBufferUsage::UnorderedAccess),
        rttr::value("ConstantBuffer", Ilum::RHIBufferUsage::ConstantBuffer)
    );
    rttr::registration::enumeration<Ilum::RHITextureUsage>("RHITextureUsage")
    (
        rttr::value("Undefined", Ilum::RHITextureUsage::Undefined),
        rttr::value("Transfer", Ilum::RHITextureUsage::Transfer),
        rttr::value("ShaderResource", Ilum::RHITextureUsage::ShaderResource),
        rttr::value("UnorderedAccess", Ilum::RHITextureUsage::UnorderedAccess),
        rttr::value("RenderTarget", Ilum::RHITextureUsage::RenderTarget)
    );
    rttr::registration::enumeration<Ilum::RHIResourceState>("RHIResourceState")
    (
        rttr::value("Undefined", Ilum::RHIResourceState::Undefined),
        rttr::value("VertexBuffer", Ilum::RHIResourceState::VertexBuffer),
        rttr::value("ConstantBuffer", Ilum::RHIResourceState::ConstantBuffer),
        rttr::value("IndexBuffer", Ilum::RHIResourceState::IndexBuffer),
        rttr::value("IndirectBuffer", Ilum::RHIResourceState::IndirectBuffer),
        rttr::value("TransferSource", Ilum::RHIResourceState::TransferSource),
        rttr::value("TransferDest", Ilum::RHIResourceState::TransferDest),
        rttr::value("ShaderResource", Ilum::RHIResourceState::ShaderResource),
        rttr::value("UnorderedAccess", Ilum::RHIResourceState::UnorderedAccess),
        rttr::value("RenderTarget", Ilum::RHIResourceState::RenderTarget),
        rttr::value("DepthWrite", Ilum::RHIResourceState::DepthWrite),
        rttr::value("DepthRead", Ilum::RHIResourceState::DepthRead),
        rttr::value("AccelerationStructure", Ilum::RHIResourceState::AccelerationStructure),
        rttr::value("Present", Ilum::RHIResourceState::Present)
    );
    rttr::registration::enumeration<Ilum::RHIFilter>("RHIFilter")
    (
        rttr::value("Nearest", Ilum::RHIFilter::Nearest),
        rttr::value("Linear", Ilum::RHIFilter::Linear)
    );
    rttr::registration::enumeration<Ilum::RHIAddressMode>("RHIAddressMode")
    (
        rttr::value("Repeat", Ilum::RHIAddressMode::Repeat),
        rttr::value("Mirrored_Repeat", Ilum::RHIAddressMode::Mirrored_Repeat),
        rttr::value("Clamp_To_Edge", Ilum::RHIAddressMode::Clamp_To_Edge),
        rttr::value("Clamp_To_Border", Ilum::RHIAddressMode::Clamp_To_Border),
        rttr::value("Mirror_Clamp_To_Edge", Ilum::RHIAddressMode::Mirror_Clamp_To_Edge)
    );
    rttr::registration::enumeration<Ilum::RHIMipmapMode>("RHIMipmapMode")
    (
        rttr::value("Nearest", Ilum::RHIMipmapMode::Nearest),
        rttr::value("Linear", Ilum::RHIMipmapMode::Linear)
    );
    rttr::registration::enumeration<Ilum::RHISamplerBorderColor>("RHISamplerBorderColor")
    (
        rttr::value("Float_Transparent_Black", Ilum::RHISamplerBorderColor::Float_Transparent_Black),
        rttr::value("Int_Transparent_Black", Ilum::RHISamplerBorderColor::Int_Transparent_Black),
        rttr::value("Float_Opaque_Black", Ilum::RHISamplerBorderColor::Float_Opaque_Black),
        rttr::value("Int_Opaque_Black", Ilum::RHISamplerBorderColor::Int_Opaque_Black),
        rttr::value("Float_Opaque_White", Ilum::RHISamplerBorderColor::Float_Opaque_White),
        rttr::value("Int_Opaque_White", Ilum::RHISamplerBorderColor::Int_Opaque_White)
    );
    rttr::registration::enumeration<Ilum::RHICompareOp>("RHICompareOp")
    (
        rttr::value("Never", Ilum::RHICompareOp::Never),
        rttr::value("Less", Ilum::RHICompareOp::Less),
        rttr::value("Equal", Ilum::RHICompareOp::Equal),
        rttr::value("Less_Or_Equal", Ilum::RHICompareOp::Less_Or_Equal),
        rttr::value("Greater", Ilum::RHICompareOp::Greater),
        rttr::value("Not_Equal", Ilum::RHICompareOp::Not_Equal),
        rttr::value("Greater_Or_Equal", Ilum::RHICompareOp::Greater_Or_Equal),
        rttr::value("Always", Ilum::RHICompareOp::Always)
    );
    rttr::registration::enumeration<Ilum::RHILogicOp>("RHILogicOp")
    (
        rttr::value("Clear", Ilum::RHILogicOp::Clear),
        rttr::value("And", Ilum::RHILogicOp::And),
        rttr::value("And_Reverse", Ilum::RHILogicOp::And_Reverse),
        rttr::value("Copy", Ilum::RHILogicOp::Copy),
        rttr::value("And_Inverted", Ilum::RHILogicOp::And_Inverted),
        rttr::value("No_Op", Ilum::RHILogicOp::No_Op),
        rttr::value("XOR", Ilum::RHILogicOp::XOR),
        rttr::value("Or", Ilum::RHILogicOp::Or),
        rttr::value("Nor", Ilum::RHILogicOp::Nor),
        rttr::value("Equivalent", Ilum::RHILogicOp::Equivalent),
        rttr::value("Invert", Ilum::RHILogicOp::Invert),
        rttr::value("Or_Reverse", Ilum::RHILogicOp::Or_Reverse),
        rttr::value("Copy_Inverted", Ilum::RHILogicOp::Copy_Inverted),
        rttr::value("Or_Inverted", Ilum::RHILogicOp::Or_Inverted),
        rttr::value("Nand", Ilum::RHILogicOp::Nand),
        rttr::value("Set", Ilum::RHILogicOp::Set)
    );
    rttr::registration::enumeration<Ilum::RHIBlendFactor>("RHIBlendFactor")
    (
        rttr::value("Zero", Ilum::RHIBlendFactor::Zero),
        rttr::value("One", Ilum::RHIBlendFactor::One),
        rttr::value("Src_Color", Ilum::RHIBlendFactor::Src_Color),
        rttr::value("One_Minus_Src_Color", Ilum::RHIBlendFactor::One_Minus_Src_Color),
        rttr::value("Dst_Color", Ilum::RHIBlendFactor::Dst_Color),
        rttr::value("One_Minus_Dst_Color", Ilum::RHIBlendFactor::One_Minus_Dst_Color),
        rttr::value("Src_Alpha", Ilum::RHIBlendFactor::Src_Alpha),
        rttr::value("One_Minus_Src_Alpha", Ilum::RHIBlendFactor::One_Minus_Src_Alpha),
        rttr::value("Dst_Alpha", Ilum::RHIBlendFactor::Dst_Alpha),
        rttr::value("One_Minus_Dst_Alpha", Ilum::RHIBlendFactor::One_Minus_Dst_Alpha),
        rttr::value("Constant_Color", Ilum::RHIBlendFactor::Constant_Color),
        rttr::value("One_Minus_Constant_Color", Ilum::RHIBlendFactor::One_Minus_Constant_Color),
        rttr::value("Constant_Alpha", Ilum::RHIBlendFactor::Constant_Alpha),
        rttr::value("One_Minus_Constant_Alpha", Ilum::RHIBlendFactor::One_Minus_Constant_Alpha),
        rttr::value("Src_Alpha_Saturate", Ilum::RHIBlendFactor::Src_Alpha_Saturate),
        rttr::value("Src1_Color", Ilum::RHIBlendFactor::Src1_Color),
        rttr::value("One_Minus_Src1_Color", Ilum::RHIBlendFactor::One_Minus_Src1_Color),
        rttr::value("Src1_Alpha", Ilum::RHIBlendFactor::Src1_Alpha),
        rttr::value("One_Minus_Src1_Alpha", Ilum::RHIBlendFactor::One_Minus_Src1_Alpha)
    );
    rttr::registration::enumeration<Ilum::RHIBlendOp>("RHIBlendOp")
    (
        rttr::value("Add", Ilum::RHIBlendOp::Add),
        rttr::value("Subtract", Ilum::RHIBlendOp::Subtract),
        rttr::value("Reverse_Subtract", Ilum::RHIBlendOp::Reverse_Subtract),
        rttr::value("Min", Ilum::RHIBlendOp::Min),
        rttr::value("Max", Ilum::RHIBlendOp::Max)
    );
    rttr::registration::enumeration<Ilum::RHICullMode>("RHICullMode")
    (
        rttr::value("None", Ilum::RHICullMode::None),
        rttr::value("Front", Ilum::RHICullMode::Front),
        rttr::value("Back", Ilum::RHICullMode::Back)
    );
    rttr::registration::enumeration<Ilum::RHIFrontFace>("RHIFrontFace")
    (
        rttr::value("Counter_Clockwise", Ilum::RHIFrontFace::Counter_Clockwise),
        rttr::value("Clockwise", Ilum::RHIFrontFace::Clockwise)
    );
    rttr::registration::enumeration<Ilum::RHIPolygonMode>("RHIPolygonMode")
    (
        rttr::value("Wireframe", Ilum::RHIPolygonMode::Wireframe),
        rttr::value("Solid", Ilum::RHIPolygonMode::Solid)
    );
    rttr::registration::enumeration<Ilum::RHIPrimitiveTopology>("RHIPrimitiveTopology")
    (
        rttr::value("Point", Ilum::RHIPrimitiveTopology::Point),
        rttr::value("Line", Ilum::RHIPrimitiveTopology::Line),
        rttr::value("Triangle", Ilum::RHIPrimitiveTopology::Triangle),
        rttr::value("Patch", Ilum::RHIPrimitiveTopology::Patch)
    );
    rttr::registration::enumeration<Ilum::RHIVertexInputRate>("RHIVertexInputRate")
    (
        rttr::value("Vertex", Ilum::RHIVertexInputRate::Vertex),
        rttr::value("Instance", Ilum::RHIVertexInputRate::Instance)
    );
    rttr::registration::enumeration<Ilum::RHILoadAction>("RHILoadAction")
    (
        rttr::value("DontCare", Ilum::RHILoadAction::DontCare),
        rttr::value("Load", Ilum::RHILoadAction::Load),
        rttr::value("Clear", Ilum::RHILoadAction::Clear)
    );
    rttr::registration::enumeration<Ilum::RHIStoreAction>("RHIStoreAction")
    (
        rttr::value("DontCare", Ilum::RHIStoreAction::DontCare),
        rttr::value("Store", Ilum::RHIStoreAction::Store)
    );
    
}
}                                 