#pragma once

#include "RHIDefinitions.hpp"

#include <memory>

namespace Ilum
{
class RHIDevice;

struct BufferDesc
{
	std::string name;

	RHIBufferUsage usage;
	RHIMemoryUsage memory;
	size_t         size;
};

REFLECTION_BEGIN(BufferDesc)
REFLECTION_PROPERTY(name)
REFLECTION_PROPERTY(usage)
REFLECTION_PROPERTY(memory)
REFLECTION_PROPERTY(size)
REFLECTION_END()

class RHIBuffer
{
  public:
	RHIBuffer(RHIDevice *device, const BufferDesc &desc);
	virtual ~RHIBuffer() = default;

	static std::unique_ptr<RHIBuffer> Create(RHIDevice *device, const BufferDesc &desc);

	const BufferDesc &GetDesc() const;

	virtual void *Map() = 0;
	virtual void  Unmap() = 0;
	virtual void  Flush(size_t offset, size_t size) = 0;

  protected:
	RHIDevice *p_device = nullptr;
	BufferDesc m_desc;
};

struct BufferStateTransition
{
	RHIBuffer *buffer;
	RHIResourceState src;
	RHIResourceState dst;
};
}        // namespace Ilum