#include "RHI/RHIDescriptor.hpp"

#include "Backend/Vulkan/Descriptor.hpp"

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
	return std::make_unique<Vulkan::Descriptor>(device, meta);
}
}        // namespace Ilum