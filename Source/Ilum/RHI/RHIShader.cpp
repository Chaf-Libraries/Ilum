#include "RHIShader.hpp"

namespace Ilum
{
RHIShader::RHIShader(RHIDevice *device, const std::string &entry_point, const std::vector<uint8_t> &source) :
    p_device(device), m_entry_point(entry_point)
{
}

const std::string &RHIShader::GetEntryPoint() const
{
	return m_entry_point;
}
}        // namespace Ilum