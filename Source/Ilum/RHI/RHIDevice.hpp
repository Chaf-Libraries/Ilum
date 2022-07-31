#pragma once

#include <Core/Window.hpp>

namespace Ilum
{
class RHIDevice
{
  public:
	RHIDevice()          = default;
	virtual ~RHIDevice() = default;

	static std::unique_ptr<RHIDevice> Create();

	virtual bool IsRayTracingSupport()          = 0;
	virtual bool IsMeshShaderSupport()          = 0;
	virtual bool IsBufferDeviceAddressSupport() = 0;
	virtual bool IsBindlessResourceSupport()    = 0;
};
}        // namespace Ilum