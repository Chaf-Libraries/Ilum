#pragma once

#include <Core/Window.hpp>

#include "RHIDefinitions.hpp"

namespace Ilum
{
class RHIDevice
{
  public:
	RHIDevice(RHIBackend backend);

	virtual ~RHIDevice() = default;

	static std::unique_ptr<RHIDevice> Create(RHIBackend backend = RHIBackend::Vulkan);

	const std::string &GetName() const;

	RHIBackend GetBackend() const;

	virtual void WaitIdle() = 0;

	virtual bool IsFeatureSupport(RHIFeature feature) = 0;

  protected:
	RHIBackend m_backend;
	std::string m_name;
};
}        // namespace Ilum