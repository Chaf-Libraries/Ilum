#pragma once

#include "RHIDefinitions.hpp"

#include <memory>

namespace Ilum
{
class RHIDevice;

STRUCT(BufferDesc, Enable)
{
	std::string    name;
	RHIBufferUsage usage;
	RHIMemoryUsage memory;

	META(Min(1))
	size_t size;

	META(Min(0))
	size_t stride;

	META(Min(0))
	size_t count;
};

class RHIBuffer
{
  public:
	RHIBuffer(RHIDevice *device, const BufferDesc &desc);

	virtual ~RHIBuffer() = default;

	const std::string GetBackend() const;

	static std::unique_ptr<RHIBuffer> Create(RHIDevice *device, const BufferDesc &desc);

	const BufferDesc &GetDesc() const;

	virtual void CopyToDevice(const void *data, size_t size, size_t offset = 0) = 0;

	virtual void CopyToHost(void *data, size_t size, size_t offset = 0) = 0;

	virtual void *Map()   = 0;
	virtual void  Unmap() = 0;

	virtual void Flush(size_t offset, size_t size) = 0;

  protected:
	RHIDevice *p_device = nullptr;
	BufferDesc m_desc;
};

struct BufferStateTransition
{
	RHIBuffer       *buffer;
	RHIResourceState src;
	RHIResourceState dst;
};
}        // namespace Ilum