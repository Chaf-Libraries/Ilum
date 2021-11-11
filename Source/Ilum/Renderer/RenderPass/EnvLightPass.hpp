#pragma once

#include "Renderer/RenderGraph/RenderPass.hpp"

namespace Ilum::pass
{
	class EnvLightPass: public TRenderPass<EnvLightPass>
	{
	  public:
	    virtual void setupPipeline(PipelineState &state) override;

	    virtual void resolveResources(ResolveState &resolve) override;

	    virtual void render(RenderPassState &state) override;
	};
}