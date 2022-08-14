#include "Shader.hpp"

namespace Ilum::DX12
{
Shader::Shader(RHIDevice *device, const std::string &entry_point, const std::vector<uint8_t> &source) :
    RHIShader(device, entry_point, source)
{
}

Shader::~Shader()
{
}

const std::vector<uint8_t> &Shader::GetDXIL()
{
	return m_dxil;
}
}        // namespace Ilum::DX12