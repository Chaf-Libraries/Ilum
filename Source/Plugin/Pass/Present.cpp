#include "IPass.hpp"

#include <Core/Core.hpp>
#include <RenderGraph/RenderGraph.hpp>
#include <RenderGraph/RenderGraphBuilder.hpp>
#include <Renderer/Renderer.hpp>

#include <imgui.h>

using namespace Ilum;

class PresentPass : public IPass<PresentPass>
{
  public:
	PresentPass() = default;

	~PresentPass() = default;

	virtual void CreateDesc(RenderPassDesc *desc)
	{
		desc->SetBindPoint(BindPoint::None)
		    .Read("Present", RenderResourceDesc::Type::Texture, RHIResourceState::ShaderResource);
	}

	virtual void CreateCallback(RenderGraph::RenderTask *task, const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer)
	{
		*task = [=](RenderGraph &render_graph, RHICommand *cmd_buffer, Variant &config, RenderGraphBlackboard& black_board) {
			auto  texture = render_graph.GetTexture(desc.resources.at("Present").handle);
			renderer->SetPresentTexture(texture);
		};
	}

	virtual void OnImGui(Variant *config)
	{
	}
};

CONFIGURATION_PASS(PresentPass)