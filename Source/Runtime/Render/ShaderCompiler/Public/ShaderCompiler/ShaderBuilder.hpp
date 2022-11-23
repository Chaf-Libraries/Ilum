#pragma once

#include <RHI/RHIContext.hpp>

namespace Ilum
{
class __declspec(dllexport) ShaderBuilder
{
  public:
	ShaderBuilder(RHIContext *context);

	~ShaderBuilder();

	RHIShader *RequireShader(const std::string &filename, const std::string &entry_point, RHIShaderStage stage, std::vector<std::string> &&macros = {}, std::vector<std::string> &&includes = {}, bool cuda = false, bool force_recompile = false);

	ShaderMeta RequireShaderMeta(RHIShader *shader) const;

  private:
	RHIContext *p_rhi_context = nullptr;

	struct Impl;

	Impl *m_impl = nullptr;
};
}        // namespace Ilum