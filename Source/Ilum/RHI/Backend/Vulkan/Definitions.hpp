#pragma once

#include "RHI/RHIDefinitions.hpp"

#include <volk.h>

#include <vk_mem_alloc.h>

#include <unordered_map>

namespace Ilum::Vulkan
{
enum class VulkanFeature
{
    DynamicRendering
};

inline static std::unordered_map<RHIFormat, VkFormat> ToVulkanFormat = {
    {RHIFormat::Undefined, VK_FORMAT_UNDEFINED},
    {RHIFormat::R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM},
    {RHIFormat::B8G8R8A8_UNORM, VK_FORMAT_B8G8R8A8_UNORM},
    {RHIFormat::R16_UINT, VK_FORMAT_R16_UINT},
    {RHIFormat::R16_SINT, VK_FORMAT_R16_SINT},
    {RHIFormat::R16_FLOAT, VK_FORMAT_R16_SFLOAT},
    {RHIFormat::R16G16_UINT, VK_FORMAT_R16G16_UINT},
    {RHIFormat::R16G16_SINT, VK_FORMAT_R16G16_SINT},
    {RHIFormat::R16G16_FLOAT, VK_FORMAT_R16G16_SFLOAT},
    {RHIFormat::R16G16B16A16_UINT, VK_FORMAT_R16G16B16A16_UINT},
    {RHIFormat::R16G16B16A16_SINT, VK_FORMAT_R16G16B16A16_SINT},
    {RHIFormat::R16G16B16A16_FLOAT, VK_FORMAT_R16G16B16A16_SFLOAT},
    {RHIFormat::R32_UINT, VK_FORMAT_R32_UINT},
    {RHIFormat::R32_SINT, VK_FORMAT_R32_SINT},
    {RHIFormat::R32_FLOAT, VK_FORMAT_R32_SFLOAT},
    {RHIFormat::R32G32_UINT, VK_FORMAT_R32G32_UINT},
    {RHIFormat::R32G32_SINT, VK_FORMAT_R32G32_SINT},
    {RHIFormat::R32G32_FLOAT, VK_FORMAT_R32G32_SFLOAT},
    {RHIFormat::R32G32B32_UINT, VK_FORMAT_R32G32B32_UINT},
    {RHIFormat::R32G32B32_SINT, VK_FORMAT_R32G32B32_SINT},
    {RHIFormat::R32G32B32_FLOAT, VK_FORMAT_R32G32B32_SFLOAT},
    {RHIFormat::R32G32B32A32_UINT, VK_FORMAT_R32G32B32A32_UINT},
    {RHIFormat::R32G32B32A32_SINT, VK_FORMAT_R32G32B32A32_SINT},
    {RHIFormat::R32G32B32A32_FLOAT, VK_FORMAT_R32G32B32A32_SFLOAT},
    {RHIFormat::D32_FLOAT, VK_FORMAT_D32_SFLOAT},
    {RHIFormat::D24_UNORM_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT},
};

inline static std::unordered_map<uint32_t, VkSampleCountFlagBits> ToVulkanSampleCountFlag = {
    {1, VK_SAMPLE_COUNT_1_BIT},
    {2, VK_SAMPLE_COUNT_2_BIT},
    {4, VK_SAMPLE_COUNT_4_BIT},
    {8, VK_SAMPLE_COUNT_8_BIT},
    {16, VK_SAMPLE_COUNT_16_BIT},
    {32, VK_SAMPLE_COUNT_32_BIT},
    {64, VK_SAMPLE_COUNT_64_BIT},
};

inline static std::unordered_map<RHITextureDimension, VkImageViewType> ToVulkanImageViewType = {
    {RHITextureDimension::Texture1D, VK_IMAGE_VIEW_TYPE_1D},
    {RHITextureDimension::Texture2D, VK_IMAGE_VIEW_TYPE_2D},
    {RHITextureDimension::Texture3D, VK_IMAGE_VIEW_TYPE_3D},
    {RHITextureDimension::TextureCube, VK_IMAGE_VIEW_TYPE_CUBE},
    {RHITextureDimension::Texture1DArray, VK_IMAGE_VIEW_TYPE_1D_ARRAY},
    {RHITextureDimension::Texture2DArray, VK_IMAGE_VIEW_TYPE_2D_ARRAY},
    {RHITextureDimension::TextureCubeArray, VK_IMAGE_VIEW_TYPE_CUBE_ARRAY},
};

inline static std::unordered_map<RHIMemoryUsage, VmaMemoryUsage> ToVmaMemoryUsage = {
    {RHIMemoryUsage::GPU_Only, VMA_MEMORY_USAGE_GPU_ONLY},
    {RHIMemoryUsage::CPU_TO_GPU, VMA_MEMORY_USAGE_CPU_TO_GPU},
    {RHIMemoryUsage::GPU_TO_CPU, VMA_MEMORY_USAGE_GPU_TO_CPU},
};

inline static std::unordered_map<RHIFilter, VkFilter> ToVulkanFilter = {
    {RHIFilter::Linear, VK_FILTER_LINEAR},
    {RHIFilter::Nearest, VK_FILTER_NEAREST},
};

inline static std::unordered_map<RHIMipmapMode, VkSamplerMipmapMode> ToVulkanMipmapMode = {
    {RHIMipmapMode::Linear, VK_SAMPLER_MIPMAP_MODE_LINEAR},
    {RHIMipmapMode::Nearest, VK_SAMPLER_MIPMAP_MODE_NEAREST},
};

inline static std::unordered_map<RHIAddressMode, VkSamplerAddressMode> ToVulkanAddressMode = {
    {RHIAddressMode::Repeat, VK_SAMPLER_ADDRESS_MODE_REPEAT},
    {RHIAddressMode::Mirrored_Repeat, VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT},
    {RHIAddressMode::Clamp_To_Edge, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE},
    {RHIAddressMode::Clamp_To_Border, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER},
    {RHIAddressMode::Mirror_Clamp_To_Edge, VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE},
};

inline static std::unordered_map<RHISamplerBorderColor, VkBorderColor> ToVulkanBorderColor = {
    {RHISamplerBorderColor::Float_Transparent_Black, VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK},
    {RHISamplerBorderColor::Int_Transparent_Black, VK_BORDER_COLOR_INT_TRANSPARENT_BLACK},
    {RHISamplerBorderColor::Float_Opaque_Black, VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK},
    {RHISamplerBorderColor::Int_Opaque_Black, VK_BORDER_COLOR_INT_OPAQUE_BLACK},
    {RHISamplerBorderColor::Float_Opaque_White, VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE},
    {RHISamplerBorderColor::Int_Opaque_White, VK_BORDER_COLOR_INT_OPAQUE_WHITE},
};

inline static std::unordered_map<RHIPrimitiveTopology, VkPrimitiveTopology> ToVulkanPrimitiveTopology = {
    {RHIPrimitiveTopology::Point, VK_PRIMITIVE_TOPOLOGY_POINT_LIST},
    {RHIPrimitiveTopology::Line, VK_PRIMITIVE_TOPOLOGY_LINE_LIST},
    {RHIPrimitiveTopology::Triangle, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST},
    {RHIPrimitiveTopology::Patch, VK_PRIMITIVE_TOPOLOGY_PATCH_LIST},
};

inline static std::unordered_map<RHIPolygonMode, VkPolygonMode> ToVulkanPolygonMode = {
    {RHIPolygonMode::Solid, VK_POLYGON_MODE_FILL},
    {RHIPolygonMode::Wireframe, VK_POLYGON_MODE_LINE},
};

inline static std::unordered_map<RHIBlendFactor, VkBlendFactor> ToVulkanBlendFactor = {
    {RHIBlendFactor::Zero, VK_BLEND_FACTOR_ZERO},
    {RHIBlendFactor::One, VK_BLEND_FACTOR_ONE},
    {RHIBlendFactor::Src_Color, VK_BLEND_FACTOR_SRC_COLOR},
    {RHIBlendFactor::One_Minus_Src_Color, VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR},
    {RHIBlendFactor::Dst_Color, VK_BLEND_FACTOR_DST_COLOR},
    {RHIBlendFactor::One_Minus_Dst_Color, VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR},
    {RHIBlendFactor::Src_Alpha, VK_BLEND_FACTOR_SRC_ALPHA},
    {RHIBlendFactor::One_Minus_Src_Alpha, VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA},
    {RHIBlendFactor::Dst_Alpha, VK_BLEND_FACTOR_DST_ALPHA},
    {RHIBlendFactor::One_Minus_Dst_Alpha, VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA},
    {RHIBlendFactor::Constant_Color, VK_BLEND_FACTOR_CONSTANT_COLOR},
    {RHIBlendFactor::One_Minus_Constant_Color, VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR},
    {RHIBlendFactor::Constant_Alpha, VK_BLEND_FACTOR_CONSTANT_ALPHA},
    {RHIBlendFactor::One_Minus_Constant_Alpha, VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA},
    {RHIBlendFactor::Src_Alpha_Saturate, VK_BLEND_FACTOR_SRC_ALPHA_SATURATE},
    {RHIBlendFactor::Src1_Color, VK_BLEND_FACTOR_SRC1_COLOR},
    {RHIBlendFactor::One_Minus_Src1_Color, VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR},
    {RHIBlendFactor::Src1_Alpha, VK_BLEND_FACTOR_SRC1_ALPHA},
    {RHIBlendFactor::One_Minus_Src1_Alpha, VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA},
};

inline static std::unordered_map<RHIBlendOp, VkBlendOp> ToVulkanBlendOp = {
    {RHIBlendOp::Add, VK_BLEND_OP_ADD},
    {RHIBlendOp::Subtract, VK_BLEND_OP_SUBTRACT},
    {RHIBlendOp::Reverse_Subtract, VK_BLEND_OP_REVERSE_SUBTRACT},
    {RHIBlendOp::Min, VK_BLEND_OP_MIN},
    {RHIBlendOp::Max, VK_BLEND_OP_MAX},
};

inline static std::unordered_map<RHILogicOp, VkLogicOp> ToVulkanLogicOp = {
    {RHILogicOp::Clear, VK_LOGIC_OP_CLEAR},
    {RHILogicOp::And, VK_LOGIC_OP_AND},
    {RHILogicOp::And_Reverse, VK_LOGIC_OP_AND_REVERSE},
    {RHILogicOp::Copy, VK_LOGIC_OP_COPY},
    {RHILogicOp::And_Inverted, VK_LOGIC_OP_AND_INVERTED},
    {RHILogicOp::No_Op, VK_LOGIC_OP_NO_OP},
    {RHILogicOp::XOR, VK_LOGIC_OP_XOR},
    {RHILogicOp::Or, VK_LOGIC_OP_OR},
    {RHILogicOp::Nor, VK_LOGIC_OP_NOR},
    {RHILogicOp::Equivalent, VK_LOGIC_OP_EQUIVALENT},
    {RHILogicOp::Invert, VK_LOGIC_OP_INVERT},
    {RHILogicOp::Or_Reverse, VK_LOGIC_OP_OR_REVERSE},
    {RHILogicOp::Copy_Inverted, VK_LOGIC_OP_COPY_INVERTED},
    {RHILogicOp::Or_Inverted, VK_LOGIC_OP_OR_INVERTED},
    {RHILogicOp::Nand, VK_LOGIC_OP_NAND},
    {RHILogicOp::Set, VK_LOGIC_OP_SET},
};

inline static std::unordered_map<RHICompareOp, VkCompareOp> ToVulkanCompareOp = {
    {RHICompareOp::Never, VK_COMPARE_OP_NEVER},
    {RHICompareOp::Less, VK_COMPARE_OP_LESS},
    {RHICompareOp::Equal, VK_COMPARE_OP_EQUAL},
    {RHICompareOp::Less_Or_Equal, VK_COMPARE_OP_LESS_OR_EQUAL},
    {RHICompareOp::Greater, VK_COMPARE_OP_GREATER},
    {RHICompareOp::Not_Equal, VK_COMPARE_OP_NOT_EQUAL},
    {RHICompareOp::Greater_Or_Equal, VK_COMPARE_OP_GREATER_OR_EQUAL},
    {RHICompareOp::Always, VK_COMPARE_OP_ALWAYS},
};

inline static std::unordered_map<RHICullMode, VkCullModeFlagBits> ToVulkanCullMode = {
    {RHICullMode::None, VK_CULL_MODE_NONE},
    {RHICullMode::Front, VK_CULL_MODE_FRONT_BIT},
    {RHICullMode::Back, VK_CULL_MODE_BACK_BIT},
};

inline static std::unordered_map<RHIFrontFace, VkFrontFace> ToVulkanFrontFace = {
    {RHIFrontFace::Clockwise, VK_FRONT_FACE_CLOCKWISE},
    {RHIFrontFace::Clockwise, VK_FRONT_FACE_COUNTER_CLOCKWISE},
};

inline static std::unordered_map<uint32_t, VkSampleCountFlagBits> ToVulkanSampleCount = {
    {1, VK_SAMPLE_COUNT_1_BIT},
    {2, VK_SAMPLE_COUNT_2_BIT},
    {4, VK_SAMPLE_COUNT_4_BIT},
    {8, VK_SAMPLE_COUNT_8_BIT},
    {16, VK_SAMPLE_COUNT_16_BIT},
    {32, VK_SAMPLE_COUNT_32_BIT},
    {64, VK_SAMPLE_COUNT_64_BIT},
};

inline static std::unordered_map<RHIShaderStage, VkShaderStageFlagBits> ToVulkanShaderStage = {
    {RHIShaderStage::Vertex, VK_SHADER_STAGE_VERTEX_BIT},
    {RHIShaderStage::TessellationControl, VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT},
    {RHIShaderStage::TessellationEvaluation, VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT},
    {RHIShaderStage::Geometry, VK_SHADER_STAGE_GEOMETRY_BIT},
    {RHIShaderStage::Fragment, VK_SHADER_STAGE_FRAGMENT_BIT},
    {RHIShaderStage::Compute, VK_SHADER_STAGE_COMPUTE_BIT},
    {RHIShaderStage::RayGen, VK_SHADER_STAGE_RAYGEN_BIT_KHR},
    {RHIShaderStage::AnyHit, VK_SHADER_STAGE_ANY_HIT_BIT_KHR},
    {RHIShaderStage::ClosestHit, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR},
    {RHIShaderStage::Miss, VK_SHADER_STAGE_MISS_BIT_KHR},
    {RHIShaderStage::Intersection, VK_SHADER_STAGE_INTERSECTION_BIT_KHR},
    {RHIShaderStage::Callable, VK_SHADER_STAGE_CALLABLE_BIT_KHR},
    {RHIShaderStage::Mesh, VK_SHADER_STAGE_MESH_BIT_NV},
    {RHIShaderStage::Task, VK_SHADER_STAGE_TASK_BIT_NV},
};

inline static std::unordered_map<RHIVertexInputRate, VkVertexInputRate> ToVulkanVertexInputRate = {
    {RHIVertexInputRate::Instance, VK_VERTEX_INPUT_RATE_INSTANCE},
    {RHIVertexInputRate::Vertex, VK_VERTEX_INPUT_RATE_VERTEX},
};

inline static std::unordered_map<RHILoadAction, VkAttachmentLoadOp> ToVulkanLoadOp = {
    {RHILoadAction::DontCare, VK_ATTACHMENT_LOAD_OP_DONT_CARE},
    {RHILoadAction::Clear, VK_ATTACHMENT_LOAD_OP_CLEAR},
    {RHILoadAction::Load, VK_ATTACHMENT_LOAD_OP_LOAD},
};

inline static std::unordered_map<RHIStoreAction, VkAttachmentStoreOp> ToVulkanStoreOp = {
    {RHIStoreAction::DontCare, VK_ATTACHMENT_STORE_OP_DONT_CARE},
    {RHIStoreAction::Store, VK_ATTACHMENT_STORE_OP_STORE},
};

inline static VkImageUsageFlags ToVulkanImageUsage(RHITextureUsage usage)
{
	VkImageUsageFlags vk_usage = 0;

	if (usage & RHITextureUsage::Transfer)
	{
		vk_usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	}
	if (usage & RHITextureUsage::ShaderResource)
	{
		vk_usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
	}
	if (usage & RHITextureUsage::UnorderedAccess)
	{
		vk_usage |= VK_IMAGE_USAGE_STORAGE_BIT;
	}

	return vk_usage;
}

inline static VkBufferUsageFlags ToVulkanBufferUsage(RHIBufferUsage usage)
{
	VkBufferUsageFlags vk_usage = 0;

	if (usage & RHIBufferUsage::AccelerationStructure)
	{
		vk_usage |= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
		            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR;
	}
	 if (usage & RHIBufferUsage::Index)
	{
		vk_usage |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
	}
	 if (usage & RHIBufferUsage::Vertex)
	{
		vk_usage |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
	}
	 if (usage & RHIBufferUsage::Indirect)
	{
		vk_usage |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
	}
	 if (usage & RHIBufferUsage::Transfer)
	{
		vk_usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
		            VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	}
	 if (usage & RHIBufferUsage::ConstantBuffer)
	{
		vk_usage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
	}
	 if (usage & RHIBufferUsage::UnorderedAccess)
	{
		vk_usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
		            VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT;
	}
	 if (usage & RHIBufferUsage::ShaderResource)
	{
		vk_usage |= VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT;
	}

	return vk_usage;
}

inline static VkShaderStageFlags ToVulkanShaderStages(RHIShaderStage stage)
{
	VkShaderStageFlags flag = 0;

	if (stage & RHIShaderStage::Vertex)
	{
		flag |= VK_SHADER_STAGE_VERTEX_BIT;
	}
	 if (stage & RHIShaderStage::Fragment)
	{
		flag |= VK_SHADER_STAGE_FRAGMENT_BIT;
	}
	 if (stage & RHIShaderStage::TessellationControl)
	{
		flag |= VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
	}
	 if (stage & RHIShaderStage::TessellationEvaluation)
	{
		flag |= VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
	}
	 if (stage & RHIShaderStage::Geometry)
	{
		flag |= VK_SHADER_STAGE_GEOMETRY_BIT;
	}
	 if (stage & RHIShaderStage::RayGen)
	{
		flag |= VK_SHADER_STAGE_RAYGEN_BIT_KHR;
	}
	 if (stage & RHIShaderStage::AnyHit)
	{
		flag |= VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
	}
	 if (stage & RHIShaderStage::ClosestHit)
	{
		flag |= VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
	}
	 if (stage & RHIShaderStage::Miss)
	{
		flag |= VK_SHADER_STAGE_MISS_BIT_KHR;
	}
	 if (stage & RHIShaderStage::Intersection)
	{
		flag |= VK_SHADER_STAGE_INTERSECTION_BIT_KHR;
	}
	 if (stage & RHIShaderStage::Callable)
	{
		flag |= VK_SHADER_STAGE_CALLABLE_BIT_KHR;
	}
	 if (stage & RHIShaderStage::Task)
	{
		flag |= VK_SHADER_STAGE_TASK_BIT_NV;
	}
	 if (stage & RHIShaderStage::Mesh)
	{
		flag |= VK_SHADER_STAGE_MESH_BIT_NV;
	}

	return flag;
}
}        // namespace Ilum::Vulkan