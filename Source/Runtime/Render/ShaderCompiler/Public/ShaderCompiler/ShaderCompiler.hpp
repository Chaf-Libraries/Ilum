#pragma once

#include <RHI/RHIDefinitions.hpp>
#include <RHI/RHIShader.hpp>

#include <string>
#include <vector>

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

class __declspec(dllexport) ShaderCompiler
{
  public:
	ShaderCompiler();

	~ShaderCompiler();

	static ShaderCompiler &GetInstance();

	std::vector<uint8_t> Compile(const ShaderDesc &desc, ShaderMeta& meta);
};

}        // namespace Ilum