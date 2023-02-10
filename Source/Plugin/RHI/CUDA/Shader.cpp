#include "Shader.hpp"

namespace Ilum::CUDA
{
Shader::Shader(RHIDevice *device, const std::string &entry_point, const std::vector<uint8_t> &source):
    RHIShader(device, entry_point, source)
{
	std::string ptx;
	ptx.resize(source.size());
	std::memcpy(ptx.data(), source.data(), source.size());

	cuModuleLoadData(&m_module, ptx.c_str());
	cuModuleGetFunction(&m_function, m_module, entry_point.c_str());
	cuModuleGetGlobal(&m_global_param, NULL, m_module, "SLANG_globalParams");
}

Shader::~Shader()
{
	cuModuleUnload(m_module);
}

CUfunction Shader::GetFunction() const
{
	return m_function;
}

CUdeviceptr Shader::GetGlobalParam() const
{
	return m_global_param;
}
}        // namespace Ilum::CUDA