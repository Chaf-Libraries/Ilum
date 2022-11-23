#include "SpirvReflection.hpp"

#include <Core/Hash.hpp>

#include <spirv_reflect.h>

namespace Ilum
{
inline static std::unordered_map<SpvReflectShaderStageFlagBits, RHIShaderStage> ShaderStageMap = {
    {SPV_REFLECT_SHADER_STAGE_VERTEX_BIT, RHIShaderStage::Vertex},
    {SPV_REFLECT_SHADER_STAGE_TESSELLATION_CONTROL_BIT, RHIShaderStage::TessellationControl},
    {SPV_REFLECT_SHADER_STAGE_TESSELLATION_EVALUATION_BIT, RHIShaderStage::TessellationEvaluation},
    {SPV_REFLECT_SHADER_STAGE_GEOMETRY_BIT, RHIShaderStage::Geometry},
    {SPV_REFLECT_SHADER_STAGE_FRAGMENT_BIT, RHIShaderStage::Fragment},
    {SPV_REFLECT_SHADER_STAGE_COMPUTE_BIT, RHIShaderStage::Compute},
    {SPV_REFLECT_SHADER_STAGE_TASK_BIT_NV, RHIShaderStage::Task},
    {SPV_REFLECT_SHADER_STAGE_MESH_BIT_NV, RHIShaderStage::Mesh},
    {SPV_REFLECT_SHADER_STAGE_RAYGEN_BIT_KHR, RHIShaderStage::RayGen},
    {SPV_REFLECT_SHADER_STAGE_ANY_HIT_BIT_KHR, RHIShaderStage::AnyHit},
    {SPV_REFLECT_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, RHIShaderStage::ClosestHit},
    {SPV_REFLECT_SHADER_STAGE_MISS_BIT_KHR, RHIShaderStage::Miss},
    {SPV_REFLECT_SHADER_STAGE_INTERSECTION_BIT_KHR, RHIShaderStage::Intersection},
    {SPV_REFLECT_SHADER_STAGE_CALLABLE_BIT_KHR, RHIShaderStage::Callable},
};

inline static std::unordered_map<SpvReflectDescriptorType, DescriptorType> DescriptorTypeMap = {
    {SPV_REFLECT_DESCRIPTOR_TYPE_SAMPLER, DescriptorType::Sampler},
    {SPV_REFLECT_DESCRIPTOR_TYPE_SAMPLED_IMAGE, DescriptorType::TextureSRV},
    {SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_IMAGE, DescriptorType::TextureUAV},
    {SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER, DescriptorType::ConstantBuffer},
    {SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER, DescriptorType::StructuredBuffer},
    {SPV_REFLECT_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, DescriptorType::AccelerationStructure},
};

inline static std::unordered_map<SpvReflectFormat, RHIFormat> FormatMap = {
    {SPV_REFLECT_FORMAT_UNDEFINED, RHIFormat::Undefined},
    {SPV_REFLECT_FORMAT_R32_UINT, RHIFormat::R32_UINT},
    {SPV_REFLECT_FORMAT_R32_SINT, RHIFormat::R32_SINT},
    {SPV_REFLECT_FORMAT_R32_SFLOAT, RHIFormat::R32_FLOAT},
    {SPV_REFLECT_FORMAT_R32G32_UINT, RHIFormat::R32G32_UINT},
    {SPV_REFLECT_FORMAT_R32G32_SINT, RHIFormat::R32G32_SINT},
    {SPV_REFLECT_FORMAT_R32G32_SFLOAT, RHIFormat::R32G32_FLOAT},
    {SPV_REFLECT_FORMAT_R32G32B32_UINT, RHIFormat::R32G32B32_UINT},
    {SPV_REFLECT_FORMAT_R32G32B32_SINT, RHIFormat::R32G32B32_SINT},
    {SPV_REFLECT_FORMAT_R32G32B32_SFLOAT, RHIFormat::R32G32B32_FLOAT},
    {SPV_REFLECT_FORMAT_R32G32B32A32_UINT, RHIFormat::R32G32B32A32_UINT},
    {SPV_REFLECT_FORMAT_R32G32B32A32_SINT, RHIFormat::R32G32B32A32_SINT},
    {SPV_REFLECT_FORMAT_R32G32B32A32_SFLOAT, RHIFormat::R32G32B32A32_FLOAT},
};

SpirvReflection &SpirvReflection::GetInstance()
{
	static SpirvReflection spirv_reflection;
	return spirv_reflection;
}

ShaderMeta SpirvReflection::Reflect(const std::vector<uint8_t> &spirv)
{
	SpvReflectShaderModule shader_module;
	spvReflectCreateShaderModule(spirv.size(), spirv.data(), &shader_module);

	ShaderMeta meta_info = {};

	// Descriptor
	for (uint32_t i = 0; i < shader_module.descriptor_binding_count; i++)
	{
		const auto &descriptor_binding = shader_module.descriptor_bindings[i];

		meta_info.descriptors.push_back(ShaderMeta::Descriptor{
			descriptor_binding.spirv_id,
		    descriptor_binding.name,
		    descriptor_binding.count,
		    descriptor_binding.set,
		    descriptor_binding.binding,
		    DescriptorTypeMap[descriptor_binding.descriptor_type],
		    ShaderStageMap[shader_module.shader_stage]});

		HashCombine(meta_info.hash,
		            descriptor_binding.name,
		            descriptor_binding.count,
		            descriptor_binding.set,
		            descriptor_binding.binding,
		            DescriptorTypeMap[descriptor_binding.descriptor_type],
		            ShaderStageMap[shader_module.shader_stage]);
	}

	// Constant
	for (uint32_t i = 0; i < shader_module.push_constant_block_count; i++)
	{
		for (uint32_t j = 0; j < shader_module.push_constant_blocks[i].member_count; j++)
		{
			auto member = shader_module.push_constant_blocks[i].members[j];

			meta_info.constants.push_back(ShaderMeta::Constant{
				member.spirv_id,
			    member.name,
			    member.size,
			    member.absolute_offset,
			    ShaderStageMap[shader_module.shader_stage]});

			HashCombine(
			    meta_info.hash,
			    member.name,
			    member.size,
			    member.absolute_offset,
			    ShaderStageMap[shader_module.shader_stage]);
		}
	}

	// Variable
	if (shader_module.shader_stage & SPV_REFLECT_SHADER_STAGE_VERTEX_BIT)
	{
		for (uint32_t i = 0; i < shader_module.input_variable_count; i++)
		{
			meta_info.inputs.push_back(ShaderMeta::Variable{
			    shader_module.input_variables[i]->spirv_id,
			    shader_module.input_variables[i]->location,
			    FormatMap[shader_module.input_variables[i]->format]
			});
		}
	}
	else if (shader_module.shader_stage & SPV_REFLECT_SHADER_STAGE_FRAGMENT_BIT)
	{
		for (uint32_t i = 0; i < shader_module.output_variable_count; i++)
		{
			meta_info.outputs.push_back(ShaderMeta::Variable{
			    shader_module.output_variables[i]->spirv_id,
			    shader_module.output_variables[i]->location,
			    FormatMap[shader_module.output_variables[i]->format]});
		}
	}

	spvReflectDestroyShaderModule(&shader_module);

	return meta_info;
}
}        // namespace Ilum