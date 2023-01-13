#include "IPass.hpp"

#include <Core/Core.hpp>
#include <RenderGraph/RenderGraph.hpp>
#include <RenderGraph/RenderGraphBuilder.hpp>
#include <Renderer/Renderer.hpp>

#include <imgui.h>

using namespace Ilum;

class Present : public RenderPass<Present>
{
  public:
	Present() = default;

	~Present() = default;

	virtual RenderPassDesc Create(size_t &handle)
	{
		RenderPassDesc desc;
		return desc.SetBindPoint(BindPoint::None)
		    .SetName("Present")
		    .SetCategory("Output")
		    .ReadTexture2D(handle++, "Present", RHIResourceState::ShaderResource);
	}

	virtual void CreateCallback(RenderGraph::RenderTask *task, const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer)
	{
		*task = [=](RenderGraph &render_graph, RHICommand *cmd_buffer, Variant &config, RenderGraphBlackboard& black_board) {
			auto  texture = render_graph.GetTexture(desc.GetPin("Present").handle);
			renderer->SetPresentTexture(texture);
		};
	}

	virtual void OnImGui(Variant *config)
	{
	}
};

CONFIGURATION_PASS(Present)