#include "RHIFrame.hpp"
#include "RHIDevice.hpp"

#include "Backend/Vulkan/Frame.hpp"
#include "Backend/DX12/Frame.hpp"
#ifdef CUDA_ENABLE
#	include "Backend/CUDA/Frame.hpp"
#endif        // CUDA_ENABLE

namespace Ilum
{
RHIFrame::RHIFrame(RHIDevice *device) :
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
#ifdef CUDA_ENABLE
		case RHIBackend::CUDA:
			return std::make_unique<CUDA::Frame>(device);
#endif        // CUDA_ENABLE
		default:
			break;
	}

	return nullptr;
}
}        // namespace Ilum