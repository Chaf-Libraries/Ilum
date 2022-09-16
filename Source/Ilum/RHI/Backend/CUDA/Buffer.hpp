#pragma once

#include "RHI/RHIBuffer.hpp"

namespace Ilum::CUDA
{
class Buffer : public RHIBuffer
{
  public:
	Buffer(RHIDevice *device, const BufferDesc &desc);

	virtual ~Buffer() override;

	virtual void CopyToDevice(const void *data, size_t size, size_t offset = 0) override;

	virtual void CopyToHost(void *data, size_t size, size_t offset) override;

	virtual void* Map() override;

	virtual void Unmap() override;

	virtual void Flush(size_t offset, size_t size) override;

	void *GetHandle() const;

	uint64_t GetDeviceAddress() const;

  private:
	void *m_handle = nullptr;
};
}        // namespace Ilum::CUDA