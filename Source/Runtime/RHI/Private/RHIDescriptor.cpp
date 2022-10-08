#include "RHI/RHIDescriptor.hpp"
#include "RHI/RHIDevice.hpp"

#include "Backend/Vulkan/Descriptor.hpp"
#include "Backend/DX12/Descriptor.hpp"
#ifdef CUDA_ENABLE
#	include "Backend/CUDA/Descriptor.hpp"
#endif        // CUDA_ENABLE

namespace Ilum
{
RHIDescriptor::RHIDescriptor(RHIDevice *device, const ShaderMeta &meta) :
    p_device(device), m_meta(meta)
{
}

const ShaderMeta &RHIDescriptor::GetShaderMeta() const
{
	return m_meta;
}

std::unique_ptr<RHIDescriptor> RHIDescriptor::Create(RHIDevice *device, const ShaderMeta &meta)
{
	switch (device->GetBackend())
	{
		case RHIBackend::Vulkan:
			return std::make_unique<Vulkan::Descriptor>(device, meta);
		case RHIBackend::DX12:
			return std::make_unique<DX12::Descriptor>(device, meta);
#ifdef CUDA_ENABLE
		case RHIBackend::CUDA:
			return std::make_unique<CUDA::Descriptor>(device, meta);
#endif        // CUDA_ENABLE
		default:
			break;
	}
#ifdef RHI_BACKEND_VULKAN
	return std::make_unique<Vulkan::Descriptor>(device, meta);
#elif defined RHI_BACKEND_DX12
	return std::make_unique<DX12::Descriptor>(device, meta);
#elif defined RHI_BACKEND_CUDA
	return std::make_unique<CUDA::Descriptor>(device, meta);
#endif        // RHI_BACKEND
}
}        // namespace Ilum