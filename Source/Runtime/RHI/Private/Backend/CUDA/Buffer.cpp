#include "Buffer.hpp"

#include <cuda_runtime.h>

namespace Ilum::CUDA
{
Buffer::Buffer(RHIDevice *device, const BufferDesc &desc) :
    RHIBuffer(device, desc)
{
	m_desc.size = m_desc.size == 0 ? m_desc.stride * m_desc.count : m_desc.size;

	if (m_desc.memory == RHIMemoryUsage::GPU_Only)
	{
		cudaMalloc(&m_handle, m_desc.size);
	}
	else
	{
		cudaMallocManaged(&m_handle, m_desc.size);
	}
}

Buffer::~Buffer()
{
	cudaFree(m_handle);
}

void Buffer::CopyToDevice(const void *data, size_t size, size_t offset)
{
	cudaMemcpy(m_handle, data, size, cudaMemcpyHostToDevice);
}

void Buffer::CopyToHost(void *data, size_t size, size_t offset)
{
	cudaMemcpy(data, m_handle, size, cudaMemcpyDeviceToHost);
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
}        // namespace Ilum