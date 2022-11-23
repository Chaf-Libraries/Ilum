#pragma once

#include <RHI/RHIShader.hpp>

namespace Ilum
{
class __declspec(dllexport) SpirvReflection
{
  public:
	static SpirvReflection &GetInstance();

	ShaderMeta Reflect(const std::vector<uint8_t> &spirv);
};
}        // namespace Ilum