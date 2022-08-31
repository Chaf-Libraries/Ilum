#include "RHIFrame.hpp"

#ifdef RHI_BACKEND_VULKAN
#	include "Backend/Vulkan/Frame.hpp"
#elif defined RHI_BACKEND_DX12
#	include "Backend/DX12/Frame.hpp"
#elif defined RHI_BACKEND_CUDA
#	include "Backend/CUDA/Frame.hpp"
#endif        // RHI_BACKEND

namespace Ilum
{
RHIFrame::RHIFrame(RHIDevice *device):
    p_device(device)
{
}

std::unique_ptr<RHIFrame> RHIFrame::Create(RHIDevice *device)
{
#ifdef RHI_BACKEND_VULKAN
	return std::make_unique<Vulkan::Frame>(device);
#elif defined RHI_BACKEND_DX12
	return std::make_unique<DX12::Frame>(device);
#elif defined RHI_BACKEND_CUDA
	return std::make_unique<CUDA::Frame>(device);
#endif        // RHI_BACKEND
	return nullptr;
}
}        // namespace Ilum