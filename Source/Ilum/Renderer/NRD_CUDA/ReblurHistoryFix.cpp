#include "ReblurHistoryFix.hpp"
#include "Renderer.hpp"

#include <RHI/RHIContext.hpp>
#include <Resource/ResourceManager.hpp>

namespace Ilum
{
RenderPassDesc ReblurHistoryFix::CreateDesc()
{
	RenderPassDesc desc = {};
	desc.SetName("ReblurHistoryFix")
	    .Read("In_Data1", RenderResourceDesc::Type::Texture, RHIResourceState::ShaderResource)
	    .Read("In_Diff", RenderResourceDesc::Type::Texture, RHIResourceState::ShaderResource)
	    .Write("Out_Diff", RenderResourceDesc::Type::Texture, RHIResourceState::UnorderedAccess)
	    .SetBindPoint(BindPoint::CUDA);

	return desc;
}

RenderGraph::RenderTask ReblurHistoryFix::Create(const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer)
{
	 auto *shader = renderer->RequireShader("Source/Shaders/NRD/ReblurHistoryFix.hlsl", "MainCS", RHIShaderStage::Compute, {}, true);

	ShaderMeta meta = renderer->RequireShaderMeta(shader);

	std::shared_ptr<RHIDescriptor>    descriptor     = std::move(renderer->GetRHIContext()->CreateDescriptor(meta, true));
	std::shared_ptr<RHIPipelineState> pipeline_state = std::move(renderer->GetRHIContext()->CreatePipelineState(true));

	pipeline_state->SetShader(RHIShaderStage::Compute, shader);

	renderer->GetResourceManager()->ImportTexture("./Asset/NRD_Data/Sample/gIn_Normal_Roughness.dds", true, false);
	auto                       *orig_In_Normal_Roughness = renderer->GetResourceManager()->GetTexture(std::to_string(Hash(std::string("./Asset/NRD_Data/Sample/gIn_Normal_Roughness.dds"), true)))->texture.get();
	std::shared_ptr<RHITexture> gfx_In_Normal_Roughness  = std::move(renderer->GetRHIContext()->CreateTexture2D(orig_In_Normal_Roughness->GetDesc().width, orig_In_Normal_Roughness->GetDesc().height, RHIFormat::R32G32B32A32_FLOAT, RHITextureUsage::Transfer, false, 1, true));
	{
		auto *cmd_buffer = renderer->GetRHIContext()->CreateCommand(RHIQueueFamily::Graphics);
		cmd_buffer->Begin();
		cmd_buffer->BlitTexture(orig_In_Normal_Roughness, TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}, RHIResourceState::ShaderResource, gfx_In_Normal_Roughness.get(), TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}, RHIResourceState::Undefined, RHIFilter::Nearest);
		cmd_buffer->End();
		renderer->GetRHIContext()->Execute(cmd_buffer);
	}
	std::shared_ptr<RHITexture> In_Normal_Roughness = renderer->GetRHIContext()->MapToCUDATexture(gfx_In_Normal_Roughness.get());

	struct
	{
		glm::mat4  gViewToClip                               = glm::mat4(glm::vec4(1.0000f, 0.0000f, 0.0000f, 0.0000f), glm::vec4(0.0000f, 1.7778f, 0.0000f, 0.0000f), glm::vec4(-0.0000f, -0.0000f, 1.0000f, 1.0000f), glm::vec4(0.0000f, 0.0000f, -0.0010f, 0.0000f));
		glm::mat4  gViewToWorld                              = glm::mat4(glm::vec4(-0.9997f, -0.0236f, -0.0000f, 0.0000f), glm::vec4(0.0031f, -0.1315f, 0.9913f, 0.0000f), glm::vec4(0.0234f, -0.9910f, -0.1315f, 0.0000f), glm::vec4(0.0000f, 0.0000f, 0.0000f, 1.0000f));
		glm::vec4  gFrustum                                  = {-1.0000f, 0.5625f, 2.0000f, -1.1250f};
		glm::vec4  gHitDistParams                            = {3.0000f, 0.1000f, 20.0000f, -25.0000f};
		glm::vec4  gViewVectorWorld                          = {-0.0234f, 0.9910f, 0.1315f, 0.0000f};
		glm::vec4  gViewVectorWorldPrev                      = {-0.0234f, 0.9910f, 0.1315f, 0.0000f};
		glm::vec2  gInvScreenSize                            = {0.0008f, 0.0014f};
		glm::vec2  gScreenSize                               = {1280.0000f, 720.0000f};
		glm::vec2  gInvRectSize                              = {0.0008f, 0.0014f};
		glm::vec2  gRectSize                                 = {1280.0000f, 720.0000f};
		glm::vec2  gRectSizePrev                             = {1280.0000f, 720.0000f};
		glm::vec2  gResolutionScale                          = {1.f, 1.f};
		glm::vec2  gRectOffset                               = {0.f, 0.f};
		glm::vec2  gSensitivityToDarkness                    = {0.01f, 0.1f};
		glm::uvec2 gRectOrigin                               = {0, 0};
		float      gReference                                = 0.f;
		float      gOrthoMode                                = 0.f;
		float      gUnproject                                = 0.0016f;
		float      gDebug                                    = 0.f;
		float      gDenoisingRange                           = 60.696f;
		float      gPlaneDistSensitivity                     = 0.005f;
		float      gFramerateScale                           = 0.5f;
		float      gBlurRadius                               = 30.f;
		float      gMaxAccumulatedFrameNum                   = 1.f;
		float      gAntiFirefly                              = 0.f;
		float      gMinConvergedStateBaseRadiusScale         = 0.25f;
		float      gLobeAngleFraction                        = 0.1f;
		float      gRoughnessFraction                        = 0.05f;
		float      gResponsiveAccumulationRoughnessThreshold = 0.f;
		float      gDiffPrepassBlurRadius                    = 30.f;
		float      gSpecPrepassBlurRadius                    = 50.f;
		uint32_t   gIsWorldSpaceMotionEnabled                = 1;
		uint32_t   gFrameIndex                               = 42372;
		uint32_t   gResetHistory                             = 0;
		uint32_t   gDiffMaterialMask                         = 1;
		uint32_t   gSpecMaterialMask                         = 0;
		float      gHistoryFixStrength                       = 1.f;
	} root_shader_parameter;
	std::shared_ptr<RHIBuffer> root_shader_parameter_buffer = std::move(renderer->GetRHIContext()->CreateBuffer<decltype(root_shader_parameter)>(1, RHIBufferUsage::ConstantBuffer, RHIMemoryUsage::CPU_TO_GPU, true));
	root_shader_parameter_buffer->CopyToDevice(&root_shader_parameter, sizeof(root_shader_parameter), 0);

	return [=](RenderGraph &render_graph, RHICommand *cmd_buffer, rttr::variant &config) {
		auto *in_data  = render_graph.GetCUDATexture(desc.resources.at("In_Data1").handle);
		auto *in_diff  = render_graph.GetCUDATexture(desc.resources.at("In_Diff").handle);
		auto *out_diff = render_graph.GetCUDATexture(desc.resources.at("Out_Diff").handle);

		 descriptor
		     ->BindBuffer("_RootShaderParameters", root_shader_parameter_buffer.get())
		     .BindSampler("gNearestClamp", renderer->GetRHIContext()->CreateSampler(SamplerDesc::NearestClamp))
		     .BindSampler("gLinearClamp", renderer->GetRHIContext()->CreateSampler(SamplerDesc::LinearClamp))
		     .BindTexture("gIn_Normal_Roughness", In_Normal_Roughness.get(), RHITextureDimension::Texture2D)
		     .BindTexture("gIn_Data1", in_data, RHITextureDimension::Texture2D)
		     .BindTexture("gIn_Diff", in_diff, RHITextureDimension::Texture2D)
		     .BindTexture("gOut_Diff", out_diff, RHITextureDimension::Texture2D);

		cmd_buffer->BindDescriptor(descriptor.get());
		cmd_buffer->BindPipelineState(pipeline_state.get());
		cmd_buffer->Dispatch(out_diff->GetDesc().width, out_diff->GetDesc().height, 1, 8, 8, 1);
	};
}
}        // namespace Ilum