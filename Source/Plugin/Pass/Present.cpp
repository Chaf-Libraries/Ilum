#include "IPass.hpp"

#include <Core/Core.hpp>
#include <RenderGraph/RenderGraph.hpp>
#include <RenderGraph/RenderGraphBuilder.hpp>
#include <Renderer/Renderer.hpp>

#include <imgui.h>

using namespace Ilum;

struct TestStruct
{
	float a;
};

class PresentPass : public IPass<PresentPass>
{
  public:
	PresentPass() = default;

	~PresentPass() = default;

	virtual void CreateDesc(RenderPassDesc *desc)
	{
		desc->SetBindPoint(BindPoint::None)
		    .SetConfig(TestStruct())
		    .Read("Present", RenderResourceDesc::Type::Texture, RHIResourceState::ShaderResource);
	}

	virtual void CreateCallback(RenderGraph::RenderTask *task, const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer)
	{
		*task = [=](RenderGraph &render_graph, RHICommand *cmd_buffer, Variant &config, RenderGraphBlackboard& black_board) {
			auto &test    = config.Convert<TestStruct>();
			auto  texture = render_graph.GetTexture(desc.resources.at("Present").handle);
			renderer->SetPresentTexture(texture);
		};
	}

	virtual void OnImGui(Variant *config)
	{
		auto &data = config->Convert<TestStruct>();
		ImGui::DragFloat("data", &data.a);
	}
};

CONFIGURATION_PASS(PresentPass)