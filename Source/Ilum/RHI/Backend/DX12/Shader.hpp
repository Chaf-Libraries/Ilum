#pragma once

#include "RHI/RHIShader.hpp"

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl.h>

using Microsoft::WRL::ComPtr;

namespace Ilum::DX12
{
class Shader : public RHIShader
{
  public:
	Shader(RHIDevice *device, const std::vector<uint8_t> &source);

	virtual ~Shader() override;

	const std::vector<uint8_t> &GetDXIL();

  private:
	std::vector<uint8_t> m_dxil;
};
}        // namespace Ilum::DX12