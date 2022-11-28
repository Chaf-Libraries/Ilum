#include "Buffer.hpp"

namespace Ilum::CUDA
{
Buffer::Buffer(RHIDevice *device, const BufferDesc &desc, HANDLE mem_handle) :
    RHIBuffer(device, desc)
{
	cudaExternalMemoryHandleDesc cuda_external_memory_handle_desc = {};

#ifdef _WIN64
	cuda_external_memory_handle_desc.type                = cudaExternalMemoryHandleTypeOpaqueWin32;
	cuda_external_memory_handle_desc.handle.win32.handle = mem_handle;
#else
	cuda_external_memory_handle_desc.type      = cudaExternalMemoryHandleTypeOpaqueFd;
	cuda_external_memory_handle_desc.handle.fd = mem_handle;
#endif
	cuda_external_memory_handle_desc.size = desc.size;

	cudaImportExternalMemory(&m_memory, &cuda_external_memory_handle_desc);

	cudaExternalMemoryBufferDesc cuda_desc = {};

	cuda_desc.size = desc.size;
	cuda_desc.offset = 0;
	cuda_desc.flags  = 0;

	cudaExternalMemoryGetMappedBuffer((void **) &m_handle, m_memory, &cuda_desc);
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
}        // namespace Ilum::CUDA