#include "Fwd.hpp"

#include "AccelerationStructure.hpp"
#include "Buffer.hpp"
#include "Command.hpp"
#include "Descriptor.hpp"
#include "Device.hpp"
#include "Frame.hpp"
#include "PipelineState.hpp"
#include "Profiler.hpp"
#include "Queue.hpp"
#include "RenderTarget.hpp"
#include "Sampler.hpp"
#include "Shader.hpp"
#include "Swapchain.hpp"
#include "Synchronization.hpp"
#include "Texture.hpp"

using namespace Ilum;
using namespace Ilum::Vulkan;

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

	EXPORT_API RHISwapchain *CreateSwapchain(Device *device, void *window_handle, uint32_t width, uint32_t height, bool vsync)
	{
		return new Swapchain(device, window_handle, width, height, vsync);
	}

	EXPORT_API RHIQueue *CreateQueue(Device *device)
	{
		return new Queue(device);
	}

	EXPORT_API RHIBuffer *CreateBuffer(Device *device, const BufferDesc &desc)
	{
		return new Buffer(device, desc);
	}

	EXPORT_API RHITexture *CreateTexture(Device *device, const TextureDesc &desc)
	{
		return new Texture(device, desc);
	}

	EXPORT_API RHISampler *CreateSampler(Device *device, const SamplerDesc &desc)
	{
		return new Sampler(device, desc);
	}

	EXPORT_API RHIShader *CreateShader(RHIDevice *device, const std::string &entry_point, const std::vector<uint8_t> &source)
	{
		return new Shader(device, entry_point, source);
	}

	EXPORT_API RHIRenderTarget *CreateRenderTarget(Device *device)
	{
		return new RenderTarget(device);
	}

	EXPORT_API RHISemaphore *CreateSemaphore(Device *device)
	{
		return new Semaphore(device);
	}

	EXPORT_API RHIFence *CreateFence(Device *device)
	{
		return new Fence(device);
	}

	EXPORT_API RHIDescriptor *CreateDescriptor(Device *device, const ShaderMeta &meta)
	{
		return new Descriptor(device, meta);
	}

	EXPORT_API RHIPipelineState *CreatePipelineState(Device *device)
	{
		return new PipelineState(device);
	}

	EXPORT_API RHIProfiler *CreateProfiler(RHIDevice *device, uint32_t frame_count)
	{
		return new Profiler(device, frame_count);
	}

	EXPORT_API RHIAccelerationStructure *CreateAccelerationStructure(Device *device)
	{
		return new AccelerationStructure(device);
	}

	EXPORT_API HANDLE GetTextureMemHandle(Device *device, Texture *texture)
	{
		HANDLE handle = {};

		VkMemoryGetWin32HandleInfoKHR handle_info = {};

		handle_info.sType  = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
		handle_info.pNext  = NULL;
		handle_info.memory = texture->GetMemory();

#ifdef _WIN64
		handle_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
		handle_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
#endif        // _WIN64

		device->GetMemoryWin32Handle(&handle_info, &handle);

		return handle;
	}

	EXPORT_API HANDLE GetBufferMemHandle(Device *device, Buffer *buffer)
	{
		HANDLE handle = {};

		VkMemoryGetWin32HandleInfoKHR handle_info = {};

		handle_info.sType  = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
		handle_info.pNext  = NULL;
		handle_info.memory = buffer->GetMemory();

#ifdef _WIN64
		handle_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
		handle_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
#endif        // _WIN64

		device->GetMemoryWin32Handle(&handle_info, &handle);

		return handle;
	}

	EXPORT_API HANDLE GetSemaphoreHandle(Device *device, Semaphore *semaphore)
	{
		HANDLE handle = {};

		VkSemaphoreGetWin32HandleInfoKHR handle_info = {};
		handle_info.sType                            = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
		handle_info.pNext                            = NULL;
		handle_info.semaphore                        = semaphore->GetHandle();

#ifdef _WIN64
		handle_info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
		handle_info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif        // _WIN64

		device->GetSemaphoreWin32Handle(&handle_info, &handle);

		return handle;
	}
}