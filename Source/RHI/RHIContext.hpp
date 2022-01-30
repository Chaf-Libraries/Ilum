#pragma once

#include <functional>
#include <memory>

namespace Ilum::RHI
{
enum class GraphicsBackend
{
	None,
	Vulkan,
	OpenGL
};

class RHIContext
{
  public:
	RHIContext() = default;

	virtual ~RHIContext(){};

	RHIContext(const RHIContext &) = delete;

	RHIContext &operator=(const RHIContext &) = delete;

	RHIContext(RHIContext &&) = delete;

	RHIContext &operator=(RHIContext &&) = delete;

	static std::shared_ptr<RHIContext> Create();

	GraphicsBackend GetGraphicsBackend() const;

	virtual float GetGPUMemoryUsed() = 0;

	virtual float GetTotalGPUMemory() = 0;

	virtual void WaitIdle() const = 0;

	virtual void OnImGui() = 0;

  private:
	static std::function<std::shared_ptr<RHIContext>(void)> CreateFunc;

  private:
#ifdef USE_VULKAN
	const GraphicsBackend m_backend = GraphicsBackend::Vulkan;
#elif USE_OPENGL
	const GraphicsBackend m_backend = GraphicsBackend::OpenGL;
#else
	const GraphicsBackend m_backend = GraphicsBackend::None;
#endif        // USE_VULKAN
};
}        // namespace Ilum::RHI