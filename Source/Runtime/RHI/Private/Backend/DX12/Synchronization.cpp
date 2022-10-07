#include "Synchronization.hpp"
#include "Device.hpp"

namespace Ilum::DX12
{
Fence::Fence(RHIDevice *device) :
    RHIFence(device)
{
	static_cast<Device *>(p_device)->GetHandle()->CreateFence(m_fence_value, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_handle));
}

void Fence::Wait(uint64_t timeout)
{
	auto val = m_handle->GetCompletedValue();
	if (m_handle->GetCompletedValue() < m_fence_value)
	{
		HANDLE event_ = CreateEvent(nullptr, FALSE, FALSE, nullptr);
		m_handle->SetEventOnCompletion(m_fence_value, event_);
		WaitForSingleObject(event_, (DWORD) timeout);
		CloseHandle(event_);
		m_fence_value++;
	}
}

void Fence::Reset()
{
}

ID3D12Fence *Fence::GetHandle()
{
	return m_handle.Get();
}

uint64_t &Fence::GetValue()
{
	return m_fence_value;
}

Semaphore::Semaphore(RHIDevice *device) :
    RHISemaphore(device)
{
}

void Semaphore::SetName(const std::string &name)
{
}
}        // namespace Ilum::DX12