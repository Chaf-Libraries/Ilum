#pragma once

#include "RHIDevice.hpp"

#include <cuda.h>

namespace Ilum::CUDA
{
class Device : public RHIDevice
{
  public:
	Device();

	~Device();

	virtual void WaitIdle() override;

	virtual bool IsFeatureSupport(RHIFeature feature) override;

  private:
	CUcontext m_context;
	CUdevice  m_device;
};
}        // namespace Ilum::CUDA