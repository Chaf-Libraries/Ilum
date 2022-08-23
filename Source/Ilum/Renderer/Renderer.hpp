#pragma once

#include <RHI/RHIContext.hpp>

//#include <RenderCore/RenderGraph/RenderGraph.hpp>

namespace Ilum
{
class Renderer
{
  public:
	Renderer(RHIContext *rhi_context);

	~Renderer();

	void Tick();

  private:
	RHIContext *p_rhi_context = nullptr;
};
}        // namespace Ilum