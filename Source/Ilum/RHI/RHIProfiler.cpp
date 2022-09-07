#include "RHIProfiler.hpp"

#ifdef RHI_BACKEND_VULKAN
#	include "Backend/Vulkan/Profiler.hpp"
#elif defined RHI_BACKEND_DX12
//#	include "Backend/DX12/Profiler.hpp"
#elif defined RHI_BACKEND_CUDA
//#	include "Backend/CUDA/Profiler.hpp"
#endif        // RHI_BACKEND

namespace Ilum
{
RHIProfiler::RHIProfiler(RHIDevice *device, uint32_t frame_count) :
    p_device(device), m_frame_count(frame_count)
{
}

std::unique_ptr<RHIProfiler> RHIProfiler::Create(RHIDevice *device, uint32_t frame_count)
{
#ifdef RHI_BACKEND_VULKAN
	return std::make_unique<Vulkan::Profiler>(device, frame_count);
#elif defined RHI_BACKEND_DX12
	return std::make_unique<DX12::Descriptor>(device, meta);
#elif defined RHI_BACKEND_CUDA
	return std::make_unique<CUDA::Descriptor>(device, meta);
#endif        // RHI_BACKEND
	return nullptr;
}

const ProfileState &RHIProfiler::GetProfileState() const
{
	return m_state;
}
}        // namespace Ilum