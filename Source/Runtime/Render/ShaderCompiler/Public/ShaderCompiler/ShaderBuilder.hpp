#pragma once

#include <RHI/RHIContext.hpp>

namespace Ilum
{
class ShaderBuilder
{
  public:
	ShaderBuilder(RHIContext *context);

	~ShaderBuilder();

	RHIShader *RequireShader(const std::string &filename, const std::string &entry_point, RHIShaderStage stage, std::vector<std::string> &&macros = {}, std::vector<std::string> &&includes = {}, bool cuda = false, bool force_recompile = false);

	ShaderMeta RequireShaderMeta(RHIShader *shader) const;

  private:
	RHIContext *p_rhi_context = nullptr;

	std::unordered_map<size_t, std::unique_ptr<RHIShader>> m_shader_cache;

	std::unordered_map<RHIShader *, ShaderMeta> m_shader_meta_cache;
};
}        // namespace Ilum