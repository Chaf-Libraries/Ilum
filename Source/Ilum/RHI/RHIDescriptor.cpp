#include "RHI/RHIDescriptor.hpp"

#ifdef RHI_BACKEND_VULKAN
#	include "Backend/Vulkan/Descriptor.hpp"
#elif defined RHI_BACKEND_DX12
#	include "Backend/DX12/Descriptor.hpp"
#elif defined RHI_BACKEND_CUDA
#	include "Backend/CUDA/Descriptor.hpp"
#endif        // RHI_BACKEND

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
#ifdef RHI_BACKEND_VULKAN
	return std::make_unique<Vulkan::Descriptor>(device, meta);
#elif defined RHI_BACKEND_DX12
	return std::make_unique<DX12::Descriptor>(device, meta);
#elif defined RHI_BACKEND_CUDA
	return std::make_unique<CUDA::Descriptor>(device, meta);
#endif        // RHI_BACKEND
}
}        // namespace Ilum