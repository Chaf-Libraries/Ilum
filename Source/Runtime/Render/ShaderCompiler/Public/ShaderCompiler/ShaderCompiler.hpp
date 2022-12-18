#pragma once

#include "Precompile.hpp"

#include <RHI/RHIDefinitions.hpp>
#include <RHI/RHIShader.hpp>

namespace Ilum
{
enum class ShaderSource
{
	HLSL,
	GLSL
};

enum class ShaderTarget
{
	SPIRV,
	DXIL,
	PTX
};

struct ShaderDesc
{
	std::string              code;
	ShaderSource             source;
	ShaderTarget             target;
	RHIShaderStage           stage;
	std::string              entry_point;
	std::vector<std::string> macros = {};
};

class EXPORT_API ShaderCompiler
{
  public:
	ShaderCompiler();

	~ShaderCompiler();

	static ShaderCompiler &GetInstance();

	std::vector<uint8_t> Compile(const ShaderDesc &desc, ShaderMeta &meta);

  private:
	struct Impl;
	Impl *m_impl = nullptr;
};

}        // namespace Ilum