#include "SpirvReflection.hpp"

#include <spirv_reflect.h>

namespace Ilum
{
ShaderMeta SpirvReflection::Reflect(const std::vector<uint8_t> &spirv)
{
	SpvReflectShaderModule shader_module;
	spvReflectCreateShaderModule(spirv.size(), spirv.data(), &shader_module);

	shader_module.spirv_execution_model;

	spvReflectDestroyShaderModule(&shader_module);

	return ShaderMeta();
}
}        // namespace Ilum