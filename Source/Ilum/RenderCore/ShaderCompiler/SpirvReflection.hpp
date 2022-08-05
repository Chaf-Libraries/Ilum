#pragma once

#include <Core/Singleton.hpp>

#include <RHI/RHIShader.hpp>

namespace Ilum
{
class SpirvReflection : public Singleton<SpirvReflection>
{
  public:
	ShaderMeta Reflect(const std::vector<uint8_t> &spirv);
};
}        // namespace Ilum