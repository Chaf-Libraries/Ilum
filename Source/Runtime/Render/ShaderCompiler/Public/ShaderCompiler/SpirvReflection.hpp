#pragma once

#include "Precompile.hpp"

#include <RHI/RHIShader.hpp>

namespace Ilum
{
class  SpirvReflection
{
  public:
	static SpirvReflection &GetInstance();

	ShaderMeta Reflect(const std::vector<uint8_t> &spirv);
};
}        // namespace Ilum