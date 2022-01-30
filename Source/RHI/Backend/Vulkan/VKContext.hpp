#pragma once

#include "../../RHIContext.hpp"

#include "Vulkan.hpp"

namespace Ilum::RHI::Vulkan
{
class VKInstance;
class VKDevice;

class VKContext : public RHIContext
{
  public:
	VKContext();

	virtual ~VKContext() override;

	virtual float GetGPUMemoryUsed() override;

	virtual float GetTotalGPUMemory() override;

	virtual void WaitIdle() const override;

	virtual void OnImGui() override;

  public:
	static VKInstance &GetInstance();

	static VKDevice &GetDevice();

  private:
	static std::unique_ptr<VKInstance> s_instance;
	static std::unique_ptr<VKDevice>   s_device;
};
}        // namespace Ilum::RHI::Vulkan