#include "RHIRenderTarget.hpp"
#include "RHIDevice.hpp"

#include <Core/Plugin.hpp>

namespace Ilum
{
RHIRenderTarget::RHIRenderTarget(RHIDevice *device) :
    p_device(device)
{
}

std::unique_ptr<RHIRenderTarget> RHIRenderTarget::Create(RHIDevice *device)
{
	return std::unique_ptr<RHIRenderTarget>(std::move(PluginManager::GetInstance().Call<RHIRenderTarget *>(fmt::format("shared/RHI/RHI.{}.dll", device->GetBackend()), "CreateRenderTarget", device)));
}

uint32_t RHIRenderTarget::GetWidth() const
{
	return m_width;
}

uint32_t RHIRenderTarget::GetHeight() const
{
	return m_height;
}

uint32_t RHIRenderTarget::GetLayers() const
{
	return m_layers;
}
}        // namespace Ilum