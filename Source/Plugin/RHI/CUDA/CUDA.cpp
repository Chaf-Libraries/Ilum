#include "Buffer.hpp"
#include "Command.hpp"
#include "Descriptor.hpp"
#include "Device.hpp"
#include "Frame.hpp"
#include "PipelineState.hpp"
#include "Profiler.hpp"
#include "Queue.hpp"
#include "Sampler.hpp"
#include "Shader.hpp"
#include "Synchronization.hpp"
#include "Texture.hpp"

#ifdef _WIN64
#	include <Windows.h>
#endif        // _WIN64

using namespace Ilum;
using namespace Ilum::CUDA;

#undef CreateSemaphore

extern "C"
{
	__declspec(dllexport) RHIDevice *CreateDevice()
	{
		return new Device;
	}

	__declspec(dllexport) RHIFrame *CreateFrame(Device *device)
	{
		return new Frame(device);
	}

	__declspec(dllexport) RHIQueue *CreateQueue(Device *device)
	{
		return new Queue(device);
	}

	__declspec(dllexport) RHITexture *MapTextureVulkanToCUDA(Device *device, const TextureDesc &desc, HANDLE mem_handle, size_t memory_size)
	{
		return new Texture(device, desc, mem_handle, memory_size);
	}

	__declspec(dllexport) RHIBuffer *MapBufferVulkanToCUDA(Device *device, const BufferDesc &desc, HANDLE mem_handle)
	{
		return new Buffer(device, desc, mem_handle);
	}

	__declspec(dllexport) RHISemaphore *MapSemaphoreVulkanToCUDA(Device *device, HANDLE handle)
	{
		return new Semaphore(device, handle);
	}

	__declspec(dllexport) RHIDescriptor *CreateDescriptor(Device *device, const ShaderMeta &meta)
	{
		return new Descriptor(device, meta);
	}
}