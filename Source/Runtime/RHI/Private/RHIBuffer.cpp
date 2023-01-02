#include "RHIBuffer.hpp"
#include "RHIDevice.hpp"

#include <Core/Plugin.hpp>

namespace Ilum
{
RHIBuffer::RHIBuffer(RHIDevice *device, const BufferDesc &desc) :
    p_device(device), m_desc(desc)
{
}

const std::string RHIBuffer::GetBackend() const
{
	return p_device->GetBackend();
}

std::unique_ptr<RHIBuffer> RHIBuffer::Create(RHIDevice *device, const BufferDesc &desc)
{
	return std::unique_ptr<RHIBuffer>(std::move(PluginManager::GetInstance().Call<RHIBuffer *>(fmt::format("shared/RHI/RHI.{}.dll", device->GetBackend()), "CreateBuffer", device, desc)));
}

const BufferDesc &RHIBuffer::GetDesc() const
{
	return m_desc;
}
}        // namespace Ilum