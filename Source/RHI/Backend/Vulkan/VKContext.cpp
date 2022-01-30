#include "VKContext.hpp"
#include "VKDevice.hpp"
#include "VKInstance.hpp"

namespace Ilum::RHI::Vulkan
{
std::unique_ptr<VKInstance> VKContext::s_instance = nullptr;
std::unique_ptr<VKDevice>   VKContext::s_device   = nullptr;

VKContext::VKContext()
{
	s_instance = std::make_unique<VKInstance>();
	s_device   = std::make_unique<VKDevice>();
}

VKContext::~VKContext()
{
}

float VKContext::GetGPUMemoryUsed()
{
	return 0.0f;
}

float VKContext::GetTotalGPUMemory()
{
	return 0.0f;
}

void VKContext::WaitIdle() const
{
	vkDeviceWaitIdle(*s_device);
}

void VKContext::OnImGui()
{
}
VKInstance &VKContext::GetInstance()
{
	return *s_instance;
}

VKDevice &VKContext::GetDevice()
{
	return *s_device;
}
}        // namespace Ilum::RHI::Vulkan

#ifdef USE_VULKAN
std::function<std::shared_ptr<Ilum::RHI::RHIContext>(void)> Ilum::RHI::RHIContext::CreateFunc =
    []() -> std::shared_ptr<Ilum::RHI::RHIContext> { return std::make_shared<Ilum::RHI::Vulkan::VKContext>(); };
#endif        // USE_VULKAN