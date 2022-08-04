#include "Shader.hpp"

namespace Ilum::DX12
{
Shader::Shader(RHIDevice *device, const std::vector<uint8_t> &source) :
    RHIShader(device, source)
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