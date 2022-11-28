#include "Fwd.hpp"

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

using namespace Ilum;
using namespace Ilum::CUDA;

#undef CreateSemaphore

extern "C"
{
	EXPORT_API RHIDevice *CreateDevice()
	{
		return new Device;
	}

	EXPORT_API RHIFrame *CreateFrame(Device *device)
	{
		return new Frame(device);
	}

	EXPORT_API RHIQueue *CreateQueue(Device *device)
	{
		return new Queue(device);
	}

	EXPORT_API RHITexture *MapTextureVulkanToCUDA(Device *device, const TextureDesc &desc, HANDLE mem_handle, size_t memory_size)
	{
		return new Texture(device, desc, mem_handle, memory_size);
	}

	EXPORT_API RHIBuffer *MapBufferVulkanToCUDA(Device *device, const BufferDesc &desc, HANDLE mem_handle)
	{
		return new Buffer(device, desc, mem_handle);
	}

	EXPORT_API RHISemaphore *MapSemaphoreVulkanToCUDA(Device *device, HANDLE handle)
	{
		return new Semaphore(device, handle);
	}

	EXPORT_API RHIDescriptor *CreateDescriptor(Device *device, const ShaderMeta &meta)
	{
		return new Descriptor(device, meta);
	}
}