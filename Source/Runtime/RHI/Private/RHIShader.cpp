#include "RHIShader.hpp"
#include "RHIDevice.hpp"

#include <Core/Plugin.hpp>

namespace Ilum
{
RHIShader::RHIShader(RHIDevice *device, const std::string &entry_point, const std::vector<uint8_t> &source) :
    p_device(device), m_entry_point(entry_point)
{
}

std::unique_ptr<RHIShader> RHIShader::Create(RHIDevice *device, const std::string &entry_point, const std::vector<uint8_t> &source)
{
	return std::unique_ptr<RHIShader>(std::move(PluginManager::GetInstance().Call<RHIShader *>(fmt::format("shared/RHI/RHI.{}.dll", device->GetBackend()), "CreateShader", device, entry_point, source)));
}

const std::string &RHIShader::GetEntryPoint() const
{
	return m_entry_point;
}
}        // namespace Ilum