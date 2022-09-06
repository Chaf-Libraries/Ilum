#pragma once

#include "RHIDefinitions.hpp"

#include <memory>

namespace Ilum
{
class RHIDevice;

struct BufferDesc
{
	std::string       name;
	RHIBufferUsage    usage;
	RHIMemoryUsage    memory;
	[[min(1)]] size_t size;
	[[min(0)]] size_t stride;
	[[min(0)]] size_t count;
};

class RHIBuffer
{
  public:
	RHIBuffer(RHIDevice *device, const BufferDesc &desc);

	virtual ~RHIBuffer() = default;

	RHIBuffer(const RHIBuffer &) = delete;
	RHIBuffer &operator=(const RHIBuffer &) = delete;

	static std::unique_ptr<RHIBuffer> Create(RHIDevice *device, const BufferDesc &desc);

	const BufferDesc &GetDesc() const;

	virtual void CopyToDevice(void *data, size_t size, size_t offset = 0) = 0;

	template <typename T>
	void CopyToDevice(T *data, size_t offset = 0)
	{
		CopyToDevice(data, sizeof(T), offset);
	}

	virtual void CopyToHost(void *data, size_t size, size_t offset) = 0;

	template <typename T>
	void CopyToHost(T *data, size_t offset = 0)
	{
		CopyToHost(data, sizeof(T), offset);
	}

	virtual void *Map()                             = 0;
	virtual void  Unmap()                           = 0;
	virtual void  Flush(size_t offset, size_t size) = 0;

  protected:
	RHIDevice *p_device = nullptr;
	BufferDesc m_desc;
};

struct [[reflection(false), serialization(false)]] BufferStateTransition
{
	RHIBuffer       *buffer;
	RHIResourceState src;
	RHIResourceState dst;
};
}        // namespace Ilum