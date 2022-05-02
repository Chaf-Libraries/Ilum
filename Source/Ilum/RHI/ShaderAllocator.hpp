#pragma once

#include "ShaderReflection.hpp"
#include "ShaderCompiler.hpp"

#include <unordered_map>

namespace Ilum
{
class RHIDevice;

class ShaderAllocator
{
  public:
	ShaderAllocator(RHIDevice* device);

	~ShaderAllocator();

	VkShaderModule Load(const ShaderDesc& desc);

	const ShaderReflectionData &Reflect(VkShaderModule shader);

  private:
	RHIDevice *p_device = nullptr;

	std::vector<VkShaderModule> m_shader_modules;
	std::vector<ShaderReflectionData> m_reflection_data;

	std::unordered_map<size_t, size_t>    m_lookup;	// hash - index
	std::unordered_map<VkShaderModule, size_t> m_mapping;
};
}