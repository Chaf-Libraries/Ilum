#include "RHIBuffer.hpp"

#include "Vulkan/Buffer.hpp"

namespace Ilum
{
RHIBuffer::RHIBuffer(RHIDevice *device, const BufferDesc &desc):
    p_device(device), m_desc(desc)
{
}

std::unique_ptr<RHIBuffer> RHIBuffer::Create(RHIDevice *device, const BufferDesc &desc)
{
	return std::make_unique<Vulkan::Buffer>(device, desc);
}

const BufferDesc &RHIBuffer::GetDesc() const
{
	return m_desc;
}
}        // namespace Ilum