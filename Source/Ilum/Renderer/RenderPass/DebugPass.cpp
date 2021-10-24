#include "DebugPass.hpp"

#include "Renderer/RenderGraph/RenderGraph.hpp"
#include "Renderer/Renderer.hpp"

#include "Graphics/Pipeline/PipelineState.hpp"

namespace Ilum::pass
{
void DebugPass::setupPipeline(PipelineState &state)
{
	RenderGraphBuilder builder;

	Renderer::instance()->buildRenderGraph(builder);
	auto &rg = builder.build();

	for (auto &[name, output] : rg->getAttachments())
	{
		if (name != rg->output())
		{
			state.addDependency(name, VK_IMAGE_USAGE_SAMPLED_BIT);
		}
	}
}
}        // namespace Ilum::pass