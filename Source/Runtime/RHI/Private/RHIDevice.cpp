#include "RHIDevice.hpp"

#include <Core/Plugin.hpp>

namespace Ilum
{
RHIDevice::RHIDevice(const std::string &backend) :
    m_backend(backend)
{
}

std::unique_ptr<RHIDevice> RHIDevice::Create(const std::string &backend)
{
	return std::unique_ptr<RHIDevice>(std::move(PluginManager::GetInstance().Call<RHIDevice *>(fmt::format("shared/RHI/RHI.{}.dll", backend), "CreateDevice")));
}

const std::string &RHIDevice::GetName() const
{
	return m_name;
}

const std::string RHIDevice::GetBackend() const
{
	return m_backend;
}
}        // namespace Ilum