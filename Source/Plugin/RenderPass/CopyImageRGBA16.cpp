#include "IPass.hpp"

#include <Core/Core.hpp>
#include <RenderGraph/RenderGraph.hpp>
#include <RenderGraph/RenderGraphBuilder.hpp>
#include <Renderer/Renderer.hpp>

#include <imgui.h>

using namespace Ilum;

class CopyImageRGBA16 : public RenderPass<CopyImageRGBA16>
{
  public:
	CopyImageRGBA16() = default;

	~CopyImageRGBA16() = default;

	virtual RenderPassDesc Create(size_t &handle)
	{
		RenderPassDesc desc;
		return desc.SetBindPoint(BindPoint::Rasterization)
		    .SetName("CopyImageRGBA16")
		    .SetCategory("Transfer")
		    .ReadTexture2D(handle++, "Input", RHIResourceState::TransferSource)
		    .WriteTexture2D(handle++, "Output", RHIFormat::R16G16B16A16_FLOAT, RHIResourceState::TransferDest);
	}

	virtual void CreateCallback(RenderGraph::RenderTask *task, const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer)
	{
		*task = [=](RenderGraph &render_graph, RHICommand *cmd_buffer, Variant &config, RenderGraphBlackboard &black_board) {
			auto input = render_graph.GetTexture(desc.GetPin("Input").handle);
			auto output = render_graph.GetTexture(desc.GetPin("Output").handle);
			cmd_buffer->BlitTexture(input, TextureRange{}, RHIResourceState::TransferSource,
			                        output, TextureRange{}, RHIResourceState::TransferDest);
		};
	}

	virtual void OnImGui(Variant *config)
	{
	}
};

CONFIGURATION_PASS(CopyImageRGBA16)