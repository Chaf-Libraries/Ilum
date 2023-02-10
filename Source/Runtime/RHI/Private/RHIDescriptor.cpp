#include "RHI/RHIDescriptor.hpp"
#include "RHI/RHIDevice.hpp"

#include <Core/Plugin.hpp>

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
	return std::unique_ptr<RHIDescriptor>(std::move(PluginManager::GetInstance().Call<RHIDescriptor *>(fmt::format("shared/RHI/RHI.{}.dll", device->GetBackend()), "CreateDescriptor", device, meta)));
}
}        // namespace Ilum