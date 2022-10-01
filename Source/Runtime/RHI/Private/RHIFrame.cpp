#include "RHIFrame.hpp"
#include "RHIDevice.hpp"

#	include "Backend/Vulkan/Frame.hpp"
#	include "Backend/DX12/Frame.hpp"
#	include "Backend/CUDA/Frame.hpp"

namespace Ilum
{
RHIFrame::RHIFrame(RHIDevice *device):
    p_device(device)
{
}

std::unique_ptr<RHIFrame> RHIFrame::Create(RHIDevice *device)
{
	switch (device->GetBackend())
	{
		case RHIBackend::Vulkan:
			return std::make_unique<Vulkan::Frame>(device);
		case RHIBackend::DX12:
			break;
		case RHIBackend::CUDA:
			return std::make_unique<CUDA::Frame>(device);
		default:
			break;
	}

	return nullptr;
}
}        // namespace Ilum