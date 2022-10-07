#pragma once

#include "RHI/RHISynchronization.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

namespace Ilum::Vulkan
{
class Device;
class Semaphore;
}        // namespace Ilum::Vulkan

namespace Ilum::CUDA
{
class Device;

class Semaphore : public RHISemaphore
{
  public:
	Semaphore(Device *device, Vulkan::Device *vk_device, Vulkan::Semaphore *vk_semaphore);

	~Semaphore();

	virtual void SetName(const std::string &name) override;

	void Signal();

	void Wait();

  private:
	Device *p_device = nullptr;

	cudaExternalSemaphore_t m_handle = nullptr;
};
}        // namespace Ilum::CUDA