#include "RHIShader.hpp"
#include "RHIDevice.hpp"

#include "Backend/Vulkan/Shader.hpp"
#include "Backend/DX12/Shader.hpp"
#include "Backend/CUDA/Shader.hpp"

namespace Ilum
{
RHIShader::RHIShader(RHIDevice *device, const std::string &entry_point, const std::vector<uint8_t> &source) :
    p_device(device), m_entry_point(entry_point)
{
}

std::unique_ptr<RHIShader> RHIShader::Create(RHIDevice *device, const std::string &entry_point, const std::vector<uint8_t> &source)
{
	switch (device->GetBackend())
	{
		case RHIBackend::Vulkan:
			return std::make_unique<Vulkan::Shader>(device, entry_point, source);
		case RHIBackend::DX12:
			return std::make_unique<DX12::Shader>(device, entry_point, source);
		case RHIBackend::CUDA:
			return std::make_unique<CUDA::Shader>(device, entry_point, source);
		default:
			break;
	}
	return nullptr;
}

const std::string &RHIShader::GetEntryPoint() const
{
	return m_entry_point;
}
}        // namespace Ilum