#include "RHIBuffer.hpp"
#include "RHIDevice.hpp"

#include "Backend/DX12/Buffer.hpp"
#include "Backend/Vulkan/Buffer.hpp"
#ifdef CUDA_ENABLE
#	include "Backend/CUDA/Buffer.hpp"
#endif        // CUDA_ENABLE

namespace Ilum
{
RHIBuffer::RHIBuffer(RHIDevice *device, const BufferDesc &desc) :
    p_device(device), m_desc(desc)
{
}

RHIBackend RHIBuffer::GetBackend() const
{
	return p_device->GetBackend();
}

std::unique_ptr<RHIBuffer> RHIBuffer::Create(RHIDevice *device, const BufferDesc &desc)
{
	switch (device->GetBackend())
	{
		case RHIBackend::Unknown:
			break;
		case RHIBackend::Vulkan:
			return std::make_unique<Vulkan::Buffer>(device, desc);
		case RHIBackend::DX12:
			return std::make_unique<DX12::Buffer>(device, desc);
#ifdef CUDA_ENABLE
		case RHIBackend::CUDA:
			return std::make_unique<CUDA::Buffer>(device, desc);
#endif        // CUDA_ENABLE
		default:
			break;
	}
	return nullptr;
}

const BufferDesc &RHIBuffer::GetDesc() const
{
	return m_desc;
}
}        // namespace Ilum