#include "RHIRenderTarget.hpp"
#include "RHIDevice.hpp"

#include "Backend/Vulkan/RenderTarget.hpp"

namespace Ilum
{
RHIRenderTarget::RHIRenderTarget(RHIDevice *device) :
    p_device(device)
{
}

std::unique_ptr<RHIRenderTarget> RHIRenderTarget::Create(RHIDevice *device)
{
	switch (device->GetBackend())
	{
		case RHIBackend::Vulkan:
			return std::make_unique<Vulkan::RenderTarget>(device);
		default:
			break;
	}
	return nullptr;
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