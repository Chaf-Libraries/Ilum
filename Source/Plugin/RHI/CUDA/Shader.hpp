#pragma once

#include "Fwd.hpp"

namespace Ilum::CUDA
{
class Shader: public RHIShader
{
  public:
	Shader(RHIDevice *device, const std::string &entry_point, const std::vector<uint8_t> &source);

	~Shader();

	CUfunction GetFunction() const;

	CUdeviceptr GetGlobalParam() const;

  private:
	CUmodule    m_module       = {};
	CUfunction  m_function     = {};
	CUdeviceptr m_global_param = {};
};
}        // namespace Ilum::CUDA