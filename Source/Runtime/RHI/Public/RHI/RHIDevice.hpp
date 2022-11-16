#pragma once

#include <Core/Window.hpp>

#include "RHIDefinitions.hpp"

namespace Ilum
{
class RHIDevice
{
  public:
	RHIDevice(const std::string &backend = "Vulkan");

	virtual ~RHIDevice() = default;

	static std::unique_ptr<RHIDevice> Create(const std::string &backend = "Vulkan");

	const std::string &GetName() const;

	const std::string GetBackend() const;

	virtual void WaitIdle() = 0;

	virtual bool IsFeatureSupport(RHIFeature feature) = 0;

  protected:
	const std::string m_backend;
	std::string m_name;
};
}        // namespace Ilum