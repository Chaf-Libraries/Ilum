#pragma once

#include "RHI/RHIBuffer.hpp"

#include <cuda_runtime.h>

namespace Ilum::Vulkan
{
class Device;
class Buffer;
}        // namespace Ilum::Vulkan

namespace Ilum::CUDA
{
class Device;

class Buffer : public RHIBuffer
{
  public:
	Buffer(Device *device, Vulkan::Device *vk_device, Vulkan::Buffer *vk_buffer);

	virtual ~Buffer() override;

	virtual void CopyToDevice(const void *data, size_t size, size_t offset = 0) override;

	virtual void CopyToHost(void *data, size_t size, size_t offset) override;

	virtual void *Map() override;

	virtual void Unmap() override;

	virtual void Flush(size_t offset, size_t size) override;

	void *GetHandle() const;

	uint64_t GetDeviceAddress() const;

  private:
	void *m_handle = nullptr;
	cudaExternalMemory_t m_memory = nullptr;
};
}        // namespace Ilum::CUDA