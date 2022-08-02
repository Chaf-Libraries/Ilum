#pragma once

#include "RHIDefinitions.hpp"

#include <memory>

namespace Ilum
{
class RHIDevice;

struct BufferDesc
{
	RHIBufferUsage usage;
	RHIMemoryUsage memory;
	size_t         size;
};

class RHIBuffer
{
  public:
	RHIBuffer(RHIDevice *device, const BufferDesc &desc);
	virtual ~RHIBuffer() = default;

	static std::unique_ptr<RHIBuffer> Create(RHIDevice *device, const BufferDesc &desc);

	const BufferDesc &GetDesc() const;

	virtual void *Map() = 0;
	virtual void  Unmap() = 0;

  protected:
	RHIDevice *p_device = nullptr;
	BufferDesc m_desc;
};

struct BufferStateTransition
{
	RHIBuffer *buffer;
	RHIBufferState src;
	RHIBufferState dst;
};
}        // namespace Ilum