#pragma once

#include "Swapchain.hpp"
#include "Device.hpp"
#include "Texture.hpp"

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl.h>

using Microsoft::WRL::ComPtr;

namespace Ilum::DX12
{
Swapchain::Swapchain(RHIDevice *device, Window *window) :
    RHISwapchain(device, window)
{
	D3D12_COMMAND_QUEUE_DESC queueDesc = {};
	queueDesc.Flags                    = D3D12_COMMAND_QUEUE_FLAG_NONE;
	queueDesc.Type                     = D3D12_COMMAND_LIST_TYPE_DIRECT;

	static_cast<Device *>(p_device)->GetHandle()->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&m_queue));

	DXGI_SWAP_CHAIN_DESC1 swapChainDesc = {};
	swapChainDesc.BufferCount           = 2;
	swapChainDesc.Width                 = window->GetWidth();
	swapChainDesc.Height                = window->GetHeight();
	swapChainDesc.Format                = DXGI_FORMAT_R8G8B8A8_UNORM;
	swapChainDesc.BufferUsage           = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	swapChainDesc.SwapEffect            = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;
	swapChainDesc.SampleDesc.Count      = 1;

	ComPtr<IDXGISwapChain1> swapchain;

	static_cast<Device *>(p_device)->GetFactory()->CreateSwapChainForHwnd(
	    m_queue.Get(),
	    (HWND) window->GetNativeHandle(),
	    &swapChainDesc,
	    nullptr,
	    nullptr,
	    &swapchain);

	static_cast<Device *>(p_device)->GetFactory()->MakeWindowAssociation((HWND) window->GetNativeHandle(), DXGI_MWA_NO_ALT_ENTER);

	swapchain.As(&m_handle);

	CreateTextures();
}

Swapchain::~Swapchain()
{
}

uint32_t Swapchain::GetTextureCount()
{
	return static_cast<uint32_t>(m_textures.size());
}

void Swapchain::AcquireNextTexture(RHISemaphore *signal_semaphore, RHIFence *signal_fence)
{
	if (m_width != p_window->GetWidth() ||
	    m_height != p_window->GetHeight())
	{
		m_width  = p_window->GetWidth();
		m_height = p_window->GetHeight();
		m_handle->ResizeBuffers(3, p_window->GetWidth(), p_window->GetHeight(), DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH);
		CreateTextures();
	}

	m_frame_index = m_handle->GetCurrentBackBufferIndex();
}

RHITexture *Swapchain::GetCurrentTexture()
{
	return m_textures[m_frame_index].get();
}

uint32_t Swapchain::GetCurrentFrameIndex()
{
	return m_frame_index;
}

void Swapchain::Present(RHISemaphore *semaphore)
{
}

void Swapchain::CreateTextures()
{
	m_textures.clear();

	{
		TextureDesc desc = {};
		desc.width       = p_window->GetWidth();
		desc.height      = p_window->GetHeight();
		desc.depth       = 1;
		desc.layers      = 1;
		desc.mips        = 1;
		desc.samples     = 1;
		desc.usage       = RHITextureUsage::RenderTarget;

		for (uint32_t i = 0; i < 3; i++)
		{
			ComPtr<ID3D12Resource> buffer;
			m_handle->GetBuffer(i, IID_PPV_ARGS(&buffer));
			m_textures.emplace_back(std::make_unique<Texture>(p_device, desc, std::move(buffer)));
		}
	}
}
}        // namespace Ilum::DX12