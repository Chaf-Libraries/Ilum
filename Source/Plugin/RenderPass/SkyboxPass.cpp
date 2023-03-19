#include "IPass.hpp"

#include <imgui.h>

using namespace Ilum;

class SkyboxPass : public RenderPass<SkyboxPass>
{
  public:
	SkyboxPass() = default;

	~SkyboxPass() = default;

	virtual RenderPassDesc Create(size_t &handle)
	{
		RenderPassDesc desc;
		return desc.SetBindPoint(BindPoint::Rasterization)
		    .SetName("SkyboxPass")
		    .SetCategory("Shading")
		    .ReadTexture2D(handle++, "Depth", RHIResourceState::DepthRead)
		    .WriteTexture2D(handle++, "Output", RHIFormat::R16G16B16A16_FLOAT, RHIResourceState::RenderTarget);
	}

	virtual void CreateCallback(RenderGraph::RenderTask *task, const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer)
	{
		auto *rhi_context    = renderer->GetRHIContext();
		auto  pipeline_state = std::shared_ptr<RHIPipelineState>(std::move(rhi_context->CreatePipelineState()));
		auto  render_target  = std::shared_ptr<RHIRenderTarget>(std::move(rhi_context->CreateRenderTarget()));

		auto vertex_shader   = renderer->RequireShader("Source/Shaders/Shading/Skybox.hlsl", "VSmain", RHIShaderStage::Vertex);
		auto fragment_shader = renderer->RequireShader("Source/Shaders/Shading/Skybox.hlsl", "PSmain", RHIShaderStage::Fragment);

		ShaderMeta shader_meta = renderer->RequireShaderMeta(vertex_shader);
		shader_meta += renderer->RequireShaderMeta(fragment_shader);

		BlendState blend_state;
		blend_state.attachment_states.resize(1);

		DepthStencilState depth_stencil_state;
		depth_stencil_state.depth_test_enable  = true;
		depth_stencil_state.depth_write_enable = false;
		depth_stencil_state.compare            = RHICompareOp::Less;

		RasterizationState rasterization_state;
		rasterization_state.cull_mode = RHICullMode::None;
		rasterization_state.front_face = RHIFrontFace::Clockwise;

		pipeline_state->SetShader(RHIShaderStage::Vertex, vertex_shader);
		pipeline_state->SetShader(RHIShaderStage::Fragment, fragment_shader);
		pipeline_state->SetBlendState(blend_state);
		pipeline_state->SetDepthStencilState(depth_stencil_state);
		pipeline_state->SetRasterizationState(rasterization_state);

		*task = [=](RenderGraph &render_graph, RHICommand *cmd_buffer, Variant &config, RenderGraphBlackboard &black_board) {
			auto depth = render_graph.GetTexture(desc.GetPin("Depth").handle);
			auto output = render_graph.GetTexture(desc.GetPin("Output").handle);

			auto *gpu_scene = black_board.Get<GPUScene>();
			auto *view = black_board.Get<View>();

			render_target->Clear();
			render_target->Set(0, output, TextureRange{}, ColorAttachment{});
			render_target->Set(depth, TextureRange{}, DepthStencilAttachment{RHILoadAction::Load});

			if (gpu_scene->texture.texture_cube)
			{
				auto descriptor = rhi_context->CreateDescriptor(shader_meta);

				descriptor->BindTexture("Skybox", gpu_scene->texture.texture_cube, RHITextureDimension::TextureCube)
				    .BindSampler("SkyboxSampler", rhi_context->CreateSampler(SamplerDesc::LinearClamp()))
					.BindBuffer("ViewBuffer", view->buffer.get());

				cmd_buffer->SetViewport(static_cast<float>(render_target->GetWidth()), static_cast<float>(render_target->GetHeight()));
				cmd_buffer->SetScissor(render_target->GetWidth(), render_target->GetHeight());
				cmd_buffer->BeginRenderPass(render_target.get());
				cmd_buffer->BindDescriptor(descriptor);
				cmd_buffer->BindPipelineState(pipeline_state.get());
				cmd_buffer->Draw(36);
				cmd_buffer->EndRenderPass();
			}
		};
	}

	virtual void OnImGui(Variant *config)
	{
	}
};

CONFIGURATION_PASS(SkyboxPass)