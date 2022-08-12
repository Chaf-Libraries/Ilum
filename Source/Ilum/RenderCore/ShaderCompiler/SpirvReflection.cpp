#include "SpirvReflection.hpp"

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

inline static std::unordered_map<SpvReflectDescriptorType, ShaderMeta::Descriptor::Type> DescriptorTypeMap = {
    {SPV_REFLECT_DESCRIPTOR_TYPE_SAMPLER, ShaderMeta::Descriptor::Type::Sampler},
    {SPV_REFLECT_DESCRIPTOR_TYPE_SAMPLED_IMAGE, ShaderMeta::Descriptor::Type::TextureSRV},
    {SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_IMAGE, ShaderMeta::Descriptor::Type::TextureUAV},
    {SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER, ShaderMeta::Descriptor::Type::ConstantBuffer},
    {SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER, ShaderMeta::Descriptor::Type::StructuredBuffer},
    {SPV_REFLECT_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, ShaderMeta::Descriptor::Type::AccelerationStructure},
};

ShaderMeta SpirvReflection::Reflect(const std::vector<uint8_t> &spirv)
{
	SpvReflectShaderModule shader_module;
	spvReflectCreateShaderModule(spirv.size(), spirv.data(), &shader_module);

	shader_module.spirv_execution_model;

	ShaderMeta meta_info = {};

	for (uint32_t i = 0; i < shader_module.descriptor_binding_count; i++)
	{
		const auto &descriptor_binding = shader_module.descriptor_bindings[i];
		meta_info.descriptors.push_back(ShaderMeta::Descriptor{
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

	spvReflectDestroyShaderModule(&shader_module);

	return meta_info;
}
}        // namespace Ilum