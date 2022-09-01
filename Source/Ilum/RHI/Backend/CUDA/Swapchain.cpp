#ifdef _WIN32
#	define WIN32_LEAN_AND_MEAN
#	define WIN32_EXTRA_LEAN
#	include <Windows.h>
#	include <Windowsx.h>
#endif        // _WIN32

#include "Swapchain.hpp"

namespace Ilum::CUDA
{
static HDC   Hdc   = nullptr;
static HGLRC Hglrc = nullptr;

Swapchain::Swapchain(RHIDevice *device, void *window, uint32_t width, uint32_t height, bool vsync) :
    RHISwapchain(device, width, height, vsync), p_window(window)
{
#ifdef _WIN32
	Hdc = GetDC((HWND) p_window);

	PIXELFORMATDESCRIPTOR pfd;
	ZeroMemory(&pfd, sizeof(PIXELFORMATDESCRIPTOR));

	pfd.nSize        = sizeof(PIXELFORMATDESCRIPTOR);
	pfd.nVersion     = 1;
	pfd.dwFlags      = PFD_SUPPORT_OPENGL | PFD_DRAW_TO_WINDOW | PFD_DOUBLEBUFFER;
	pfd.iPixelType   = PFD_TYPE_RGBA;
	pfd.cColorBits   = 24;
	pfd.cDepthBits   = 32;
	pfd.cStencilBits = 8;
	pfd.iLayerType   = PFD_MAIN_PLANE;

	int pixelFormat = ChoosePixelFormat(Hdc, &pfd);
	SetPixelFormat(Hdc, pixelFormat, &pfd);

	HGLRC hglrc = wglCreateContext(Hdc);
	wglMakeCurrent(Hdc, hglrc);

	if (!gladLoadGL())
	{
		LOG_ERROR("Failed to load OpenGL!");
	}
#endif        // _WIN32

	// glGenBuffers(1, &m_gl_handle);
	// glBindBuffer(GL_ARRAY_BUFFER, m_gl_handle);
	// unsigned int size = m_width * m_height * 4 * sizeof(float);
	// glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	// glBindBuffer(GL_ARRAY_BUFFER, 0);
	// cudaGraphicsGLRegisterBuffer(&m_cuda_handle, m_gl_handle, cudaGraphicsMapFlagsWriteDiscard);

	glCreateRenderbuffers(2, &m_gl_framebuffers[0]);
	glCreateFramebuffers(2, &m_gl_renderbuffers[0]);

	glNamedFramebufferRenderbuffer(m_gl_framebuffers[0], GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, m_gl_renderbuffers[0]);
	glNamedFramebufferRenderbuffer(m_gl_framebuffers[1], GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, m_gl_renderbuffers[1]);

	m_cuda_handles[0] = nullptr;
	m_cuda_handles[1] = nullptr;

	m_textures[0] = nullptr;
	m_textures[1] = nullptr;

	cudaStreamCreateWithFlags(&m_stream, cudaStreamDefault);

	Resize(m_width, m_height);
}

Swapchain::~Swapchain()
{
#ifdef _WIN32
	wglMakeCurrent(NULL, NULL);
	wglDeleteContext(Hglrc);
	ReleaseDC((HWND) p_window, Hdc);

	Hdc   = nullptr;
	Hglrc = nullptr;
#endif        // _WIN32

	for (uint32_t i = 0; i < 2; i++)
	{
		if (m_cuda_handles[i] != nullptr)
		{
			cudaGraphicsUnregisterResource(m_cuda_handles[i]);
		}
	}

	glDeleteRenderbuffers(2, m_gl_renderbuffers);
	glDeleteFramebuffers(2, m_gl_framebuffers);
}

uint32_t Swapchain::GetTextureCount()
{
	return 2;
}

void Swapchain::AcquireNextTexture(RHISemaphore *signal_semaphore, RHIFence *signal_fence)
{
	glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	m_frame_index = (m_frame_index + 1) % 2;
	cudaGraphicsMapResources(1, &m_cuda_handles[m_frame_index], m_stream);
}

RHITexture *Swapchain::GetCurrentTexture()
{
	return m_textures[m_frame_index].get();
}

uint32_t Swapchain::GetCurrentFrameIndex()
{
	return m_frame_index;
}

bool Swapchain::Present(RHISemaphore *semaphore)
{
	//// Unmap back buffer
	// cudaGraphicsUnmapResources(1, &m_cuda_handle, 0);

	//// Draw back buffer
	// glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	// glBindBuffer(GL_ARRAY_BUFFER, m_gl_handle);
	// glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *) 0);
	// glEnableVertexAttribArray(0);
	// glDrawArrays(GL_POINTS, 0, m_width * m_height);
	// glDisableVertexAttribArray(0);

	cudaGraphicsUnmapResources(1, &m_cuda_handles[m_frame_index], m_stream);

	glBlitNamedFramebuffer(m_gl_framebuffers[m_frame_index], 0, 0, 0, m_width, m_height, 0, m_height,
	                       m_width, 0, GL_COLOR_BUFFER_BIT, GL_NEAREST);

	 SwapBuffers(Hdc);
	return true;
}

void Swapchain::Resize(uint32_t width, uint32_t height)
{
	if (width == 0 || height == 0)
	{
		return;
	}

	m_width  = width;
	m_height = height;

	for (uint32_t i = 0; i < 2; i++)
	{
		if (m_cuda_handles[i] != nullptr)
		{
			cudaGraphicsUnregisterResource(m_cuda_handles[i]);
		}

		glNamedRenderbufferStorage(m_gl_renderbuffers[i], GL_RGBA32F, m_width, m_height);

		cudaGraphicsGLRegisterImage(&m_cuda_handles[i], m_gl_renderbuffers[i], GL_RENDERBUFFER, cudaGraphicsRegisterFlagsSurfaceLoadStore | cudaGraphicsRegisterFlagsWriteDiscard);
	}

	cudaGraphicsMapResources(2, &m_cuda_handles[0], 0);
	for (uint32_t index = 0; index < 2; index++)
	{
		cudaArray_t cuda_array = nullptr;
		cudaGraphicsSubResourceGetMappedArray(&cuda_array, m_cuda_handles[index], 0, 0);
		TextureDesc desc  = {};
		desc.name         = "Swapchain Image " + std::to_string(index);
		desc.width        = m_width;
		desc.height       = m_height;
		desc.depth        = 1;
		desc.layers       = 1;
		desc.mips         = 1;
		desc.samples      = 1;
		desc.format       = RHIFormat::R32G32B32A32_FLOAT;
		desc.usage        = RHITextureUsage::RenderTarget | RHITextureUsage::UnorderedAccess | RHITextureUsage::ShaderResource;
		m_textures[index] = std::make_unique<Texture>(p_device, cuda_array, desc);
	}
	cudaGraphicsUnmapResources(2, &m_cuda_handles[0], 0);
}
}        // namespace Ilum::CUDA