#include "RHIProfiler.hpp"
#include "RHIDevice.hpp"

#include "Backend/CUDA/Profiler.hpp"
#include "Backend/DX12/Profiler.hpp"
#include "Backend/Vulkan/Profiler.hpp"

namespace Ilum
{
RHIProfiler::RHIProfiler(RHIDevice *device, uint32_t frame_count) :
    p_device(device), m_frame_count(frame_count)
{
}

std::unique_ptr<RHIProfiler> RHIProfiler::Create(RHIDevice *device, uint32_t frame_count)
{
	switch (device->GetBackend())
	{
		case RHIBackend::Vulkan:
			return std::make_unique<Vulkan::Profiler>(device, frame_count);
		case RHIBackend::DX12:
			return nullptr;
		case RHIBackend::CUDA:
			return std::make_unique<CUDA::Profiler>(device, frame_count);
		default:
			break;
	}
	return nullptr;
}

const ProfileState &RHIProfiler::GetProfileState() const
{
	return m_state;
}
}        // namespace Ilum