#pragma once

#include "RHI/RHISwapchain.hpp"

#include <directx/d3d12.h>
#include <dxgi1_6.h>
#include <wrl.h>

using Microsoft::WRL::ComPtr;

namespace Ilum::DX12
{
class Texture;

class Swapchain : public RHISwapchain
{
  public:
	Swapchain(RHIDevice *device, Window *window);
	virtual ~Swapchain() override;

	virtual uint32_t GetTextureCount() override;

	virtual void AcquireNextTexture(RHISemaphore *signal_semaphore, RHIFence *signal_fence) override;

	virtual RHITexture *GetCurrentTexture() override;

	virtual uint32_t GetCurrentFrameIndex() override;

	virtual void Present(RHISemaphore *semaphore) override;

  private:
	void CreateTextures();

  private:
	std::vector<std::unique_ptr<Texture>> m_textures;
	ComPtr<ID3D12CommandQueue>            m_queue  = nullptr;
	ComPtr<IDXGISwapChain3>               m_handle = nullptr;
	ComPtr<ID3D12Fence>                   m_fence  = nullptr;

	std::vector<uint64_t> m_fence_value;
	HANDLE                m_fence_event = nullptr;

	uint32_t m_width  = 0;
	uint32_t m_height = 0;

	uint32_t m_frame_index = 0;
};
}        // namespace Ilum::DX12