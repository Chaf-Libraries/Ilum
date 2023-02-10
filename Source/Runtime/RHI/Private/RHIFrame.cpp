#include "RHIFrame.hpp"
#include "RHIDevice.hpp"

#include <Core/Plugin.hpp>

namespace Ilum
{
RHIFrame::RHIFrame(RHIDevice *device) :
    p_device(device)
{
}

std::unique_ptr<RHIFrame> RHIFrame::Create(RHIDevice *device)
{
	return std::unique_ptr<RHIFrame>(std::move(PluginManager::GetInstance().Call<RHIFrame *>(fmt::format("shared/RHI/RHI.{}.dll", device->GetBackend()), "CreateFrame", device)));
}
}        // namespace Ilum