#include "RHIFrame.hpp"

#include "Backend/Vulkan/Frame.hpp"

namespace Ilum
{
RHIFrame::RHIFrame(RHIDevice *device):
    p_device(device)
{
}

std::unique_ptr<RHIFrame> RHIFrame::Create(RHIDevice *device)
{
	return std::make_unique<Vulkan::Frame>(device);
}
}        // namespace Ilum