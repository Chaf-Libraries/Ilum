#pragma once

#include "RHIDefinitions.hpp"

#include <memory>

namespace Ilum
{
class RHIDevice;

REFLECTION_STRUCT BufferDesc
{
	REFLECTION_PROPERTY()
	std::string name;

	REFLECTION_PROPERTY()
	RHIBufferUsage usage;

	REFLECTION_PROPERTY()
	RHIMemoryUsage memory;

	REFLECTION_PROPERTY()
	size_t size;
};

class RHIBuffer
{
  public:
	RHIBuffer(RHIDevice *device, const BufferDesc &desc);
	virtual ~RHIBuffer() = default;

	static std::unique_ptr<RHIBuffer> Create(RHIDevice *device, const BufferDesc &desc);

	const BufferDesc &GetDesc() const;

	virtual void *Map()                             = 0;
	virtual void  Unmap()                           = 0;
	virtual void  Flush(size_t offset, size_t size) = 0;

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