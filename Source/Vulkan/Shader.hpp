#pragma once

#include "Vulkan.hpp"

#include "ShaderCompiler/SpirvReflection.hpp"

#include <map>

namespace Ilum::Vulkan
{
class Shader
{
  public:
	Shader(const std::vector<uint32_t> &spirv, VkShaderStageFlagBits stage);
	~Shader();

	Shader(const Shader &) = delete;
	Shader &operator=(const Shader &) = delete;
	Shader(Shader &&)                 = delete;
	Shader &operator=(Shader &&) = delete;

	operator const VkShaderModule &() const;

	const VkShaderModule &GetHandle() const;

	VkShaderStageFlagBits GetStage() const;

	const ReflectionData &GetReflectionData() const;

  private:
	VkShaderModule        m_handle = VK_NULL_HANDLE;
	VkShaderStageFlagBits m_stage;
	ReflectionData        m_reflection_data;
};

class ShaderCache
{
  public:
	ShaderCache()  = default;
	~ShaderCache() = default;

	const Shader &RequestShader(const std::string &path, VkShaderStageFlagBits stage);
  private:
	std::map<size_t, std::unique_ptr<Shader>> m_shaders;
};
}        // namespace Ilum::Vulkan