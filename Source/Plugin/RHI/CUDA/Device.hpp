#pragma once

#include <RHI/RHIDevice.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

namespace Ilum::CUDA
{
class Device : public RHIDevice
{
  public:
	Device();

	~Device();

	virtual void WaitIdle() override;

	virtual bool IsFeatureSupport(RHIFeature feature) override;

	cudaStream_t GetSteam() const;

  private:
	CUcontext m_context;
	CUdevice  m_device;
	cudaStream_t m_steam;
};
}        // namespace Ilum::CUDA