#pragma once

#include "Renderer/RenderGraph/RenderPass.hpp"

namespace Ilum::pass
{
	class DebugPass : public TRenderPass<DebugPass>
	{
	  public:
	    DebugPass() = default;

		~DebugPass() = default;

		virtual void setupPipeline(PipelineState &state);
	};
}