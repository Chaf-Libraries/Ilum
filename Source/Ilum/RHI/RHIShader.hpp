#pragma once

#include <vector>

namespace Ilum
{
class RHIDevice;

struct ShaderMeta
{
};

class RHIShader
{
  public:
	RHIShader(RHIDevice *device, const std::vector<uint8_t> &source);

	virtual ~RHIShader() = 0;

  protected:
	RHIDevice *p_device = nullptr;
};
}        // namespace Ilum