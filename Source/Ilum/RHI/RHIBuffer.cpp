#include "RHIBuffer.hpp"

#ifdef RHI_BACKEND_VULKAN
#	include "Backend/Vulkan/Buffer.hpp"
#elif defined RHI_BACKEND_DX12
#	include "Backend/DX12/Buffer.hpp"
#endif        // RHI_BACKEND

namespace Ilum
{
RHIBuffer::RHIBuffer(RHIDevice *device, const BufferDesc &desc):
    p_device(device), m_desc(desc)
{
}

std::unique_ptr<RHIBuffer> RHIBuffer::Create(RHIDevice *device, const BufferDesc &desc)
{
#ifdef RHI_BACKEND_VULKAN
	return std::make_unique<Vulkan::Buffer>(device, desc);
#elif defined RHI_BACKEND_DX12
	return std::make_unique<DX12::Buffer>(device, desc);
#endif
	return nullptr;
}

const BufferDesc &RHIBuffer::GetDesc() const
{
	return m_desc;
}
}        // namespace Ilum