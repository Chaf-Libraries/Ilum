#include "RHIShader.hpp"

#ifdef RHI_BACKEND_VULKAN
#	include "Backend/Vulkan/Shader.hpp"
#elif defined RHI_BACKEND_DX12
#	include "Backend/DX12/Shader.hpp"
#elif defined RHI_BACKEND_CUDA
#	include "Backend/CUDA/Shader.hpp"
#endif        // RHI_BACKEND

namespace Ilum
{
RHIShader::RHIShader(RHIDevice *device, const std::string &entry_point, const std::vector<uint8_t> &source) :
    p_device(device), m_entry_point(entry_point)
{
}

std::unique_ptr<RHIShader> RHIShader::Create(RHIDevice *device, const std::string &entry_point, const std::vector<uint8_t> &source)
{
#ifdef RHI_BACKEND_VULKAN
	return std::make_unique<Vulkan::Shader>(device, entry_point, source);
#elif defined RHI_BACKEND_DX12
	return std::make_unique<DX12::Shader>(device, entry_point, source);
#elif defined RHI_BACKEND_CUDA
	return std::make_unique<CUDA::Shader>(device, entry_point, source);
#endif        // RHI_BACKEND
}

const std::string &RHIShader::GetEntryPoint() const
{
	return m_entry_point;
}
}        // namespace Ilum