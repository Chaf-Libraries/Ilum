#pragma once

#include "Shader.hpp"

#include <map>

namespace Ilum::Graphics
{
class Device;

class ShaderCache
{
  public:
	ShaderCache(const Device& device);
	~ShaderCache() = default;

	const Shader &RequestShader(const std::string &path, VkShaderStageFlagBits stage);

  private:
	const Device &                            m_device;
	std::map<size_t, std::unique_ptr<Shader>> m_shaders;
};
}        // namespace Ilum::Graphics