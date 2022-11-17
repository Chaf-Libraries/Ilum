#include "RenderPass.hpp"

#include <RenderGraph/RenderGraph.hpp>
#include <RenderGraph/RenderGraphBuilder.hpp>
#include <Renderer/Renderer.hpp>
#include <iostream>

using namespace Ilum;

struct TestStruct
{
	float a;
};

extern "C"
{
	__declspec(dllexport) void CreateDesc(RenderPassDesc *desc)
	{
		desc->SetBindPoint(BindPoint::None)
		    .SetConfig(TestStruct())
		    .Read("Present", RenderResourceDesc::Type::Texture, RHIResourceState::ShaderResource);
	}

	__declspec(dllexport) void CreateCallback(RenderGraph::RenderTask *task, const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer)
	{
		*task = [=](RenderGraph &render_graph, RHICommand *cmd_buffer, rttr::variant &config) {
			auto test    = config.convert<TestStruct>();
			auto texture = render_graph.GetTexture(desc.resources.at("Present").handle);
			renderer->SetPresentTexture(texture);
		};
	}
}