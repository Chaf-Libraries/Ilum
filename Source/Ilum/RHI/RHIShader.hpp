#pragma once

#include <vector>

namespace Ilum
{
struct ShaderMeta
{
};

class RHIShader
{
  public:
	RHIShader(const std::vector<uint8_t> &source);
	virtual ~RHIShader() = 0;
};
}        // namespace Ilum