#include "Buffer.hpp"

#include <cuda_runtime.h>

namespace Ilum::CUDA
{
Buffer::Buffer(RHIDevice *device, const BufferDesc &desc):
    RHIBuffer(device, desc)
{
	if (m_desc.memory == RHIMemoryUsage::GPU_Only)
	{
		cudaMalloc(&m_handle, desc.size);
	}
	else
	{
		cudaMallocManaged(&m_handle, desc.size);
	}
}

Buffer::~Buffer()
{
	cudaFree(m_handle);
}

void *Buffer::Map()
{
	if (m_desc.memory != RHIMemoryUsage::GPU_Only)
	{
		return m_handle;
	}
	return nullptr;
}

void Buffer::Unmap()
{

}

void Buffer::Flush(size_t offset, size_t size)
{

}

void *Buffer::GetHandle() const
{
	return m_handle;
}

uint64_t Buffer::GetDeviceAddress() const
{
	return (uint64_t) m_handle;
}
}        // namespace Ilum::CUDA