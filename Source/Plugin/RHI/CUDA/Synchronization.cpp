#include "Synchronization.hpp"
#include "Device.hpp"

#include <volk.h>

namespace Ilum::CUDA
{

Semaphore::Semaphore(RHIDevice *device, HANDLE handle) :
    RHISemaphore(device)
{
	cudaExternalSemaphoreHandleDesc external_semaphore_handle_desc = {};

#ifdef _WIN64
	external_semaphore_handle_desc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32;

	external_semaphore_handle_desc.handle.win32.handle = handle;
#else
	external_semaphore_handle_desc.type      = cudaExternalSemaphoreHandleTypeOpaqueFd;
	external_semaphore_handle_desc.handle.fd = handle;
#endif
	external_semaphore_handle_desc.flags = 0;

	cudaImportExternalSemaphore(&m_handle, &external_semaphore_handle_desc);
}

Semaphore::~Semaphore()
{
	cudaDestroyExternalSemaphore(m_handle);
}

void Semaphore::SetName(const std::string &name)
{
}

void Semaphore::Signal()
{
	cudaExternalSemaphoreSignalParams params = {};

	params.params.fence.value = 0;
	params.flags              = 0;

	cudaSignalExternalSemaphoresAsync(&m_handle, &params, 1, static_cast<Device*>(p_device)->GetSteam());
}

void Semaphore::Wait()
{
	cudaExternalSemaphoreWaitParams params = {};

	params.params.fence.value = 0;
	params.flags              = 0;

	cudaWaitExternalSemaphoresAsync(&m_handle, &params, 1, static_cast<Device *>(p_device)->GetSteam());
}
}        // namespace Ilum::CUDA