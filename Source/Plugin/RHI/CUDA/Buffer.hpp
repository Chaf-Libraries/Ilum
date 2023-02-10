#pragma once

#include "Fwd.hpp"

namespace Ilum::CUDA
{
class Buffer : public RHIBuffer
{
  public:
	Buffer(RHIDevice *device, const BufferDesc &desc, HANDLE mem_handle);

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