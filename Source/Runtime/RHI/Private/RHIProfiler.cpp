#include "RHIProfiler.hpp"
#include "RHIDevice.hpp"

#include <Core/Plugin.hpp>

namespace Ilum
{
RHIProfiler::RHIProfiler(RHIDevice *device, uint32_t frame_count) :
    p_device(device), m_frame_count(frame_count)
{
}

std::unique_ptr<RHIProfiler> RHIProfiler::Create(RHIDevice *device, uint32_t frame_count)
{
	return std::unique_ptr<RHIProfiler>(std::move(PluginManager::GetInstance().Call<RHIProfiler *>(fmt::format("RHI.{}.dll", device->GetBackend()), "CreateProfiler", device, frame_count)));
}

const ProfileState &RHIProfiler::GetProfileState() const
{
	return m_state;
}
}        // namespace Ilum