#include "RHIDevice.hpp"
#include "Vulkan/Device.hpp"

namespace Ilum
{
std::unique_ptr<RHIDevice> RHIDevice::Create()
{
	return std::make_unique<Vulkan::Device>();
}
}        // namespace Ilum