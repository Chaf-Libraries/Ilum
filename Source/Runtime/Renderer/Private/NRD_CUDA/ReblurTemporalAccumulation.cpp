#pragma once

#include "ReblurTemporalAccumulation.hpp"

#include <RHI/RHIContext.hpp>
#include <Resource/ResourceManager.hpp>

namespace Ilum
{
RenderPassDesc ReblurTemporalAccumulation::CreateDesc()
{
	RenderPassDesc desc = {};
	desc.SetName("ReblurTemporalAccumulation")
	    .Write("Out_Diff", RenderResourceDesc::Type::Texture, RHIResourceState::UnorderedAccess)
	    .Write("Out_Data1", RenderResourceDesc::Type::Texture, RHIResourceState::UnorderedAccess)
	    .SetBindPoint(BindPoint::CUDA);

	return desc;
}

RenderGraph::RenderTask ReblurTemporalAccumulation::Create(const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer)
{
	auto *shader = renderer->RequireShader("Source/Shaders/NRD/ReblurTemporalAccumulation.hlsl", "MainCS", RHIShaderStage::Compute, {}, true);

	ShaderMeta meta = renderer->RequireShaderMeta(shader);

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
		glm::vec2  gSensitivityToDarkness                    = {0.0100f, 0.1000f};
		glm::uvec2 gRectOrigin                               = {0, 0};
		float      gReference                                = 0.f;
		float      gOrthoMode                                = 0.f;
		float      gUnproject                                = 0.0016f;
		float      gDebug                                    = 0.f;
		float      gDenoisingRange                           = 60.696f;
		float      gPlaneDistSensitivity                     = 0.005f;
		float      gFramerateScale                           = 0.5f;
		float      gBlurRadius                               = 30.f;
		float      gMaxAccumulatedFrameNum                   = 0.f;
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
		glm::mat4  gWorldToViewPrev                          = glm::mat4(glm::vec4(-0.9997f, 0.0031f, 0.0234f, 0.0000f), glm::vec4(-0.0236f, -0.1315f, -0.9910f, 0.0000f), glm::vec4(-0.0000f, 0.9913f, -0.1315f, 0.0000f), glm::vec4(0.0000f, -0.0000f, -0.0000f, 1.0000f));
		glm::mat4  gWorldToClipPrev                          =glm::mat4(glm::vec4(-0.9997f, 0.0055f, 0.0234f, 0.0234f), glm::vec4(-0.0236f, -0.2337f, -0.9910f, -0.9910f), glm::vec4(0.0000f, 1.7623f, -0.1315f, -0.1315f), glm::vec4(0.0000f, 0.0000f, -0.0010f, 0.0000f));
		glm::mat4  gWorldToClip                              = glm::mat4(glm::vec4(-0.9997f, 0.0055f, 0.0234f, 0.0234f), glm::vec4(-0.0236f, -0.2337f, -0.9910f, -0.9910f), glm::vec4(0.0000f, 1.7623f, -0.1315f, -0.1315f), glm::vec4(0.0000f, 0.0000f, -0.0010f, 0.0000f));
		glm::mat4  gWorldPrevToWorld                         = glm::mat4(glm::vec4(1.0000f, 0.0000f, 0.0000f, 0.0000f), glm::vec4(0.0000f, 1.0000f, 0.0000f, 0.0000f), glm::vec4(0.0000f, 0.0000f, 1.0000f, 0.0000f), glm::vec4(0.0000f, 0.0000f, 0.0000f, 1.0000f));
		glm::vec4  gFrustumPrev                              = {-1.0000f, 0.5625f, 2.0000f, -1.1250f};
		glm::vec4  gCameraDelta                              = {0.0000f, 0.0000f, 0.0000f, 0.0000f};
		glm::vec4  gRotator                                  = {-0.5868f, 0.8098f, -0.8098f, -0.5868f};
		glm::vec2  gMotionVectorScale                        = {1.f, 1.f};
		float      gCheckerboardResolveAccumSpeed            = 0.6447f;
		float      gDisocclusionThreshold                    = 0.0071f;
		uint32_t   gDiffCheckerboard                         = 2;
		uint32_t   gSpecCheckerboard                         = 2;
		uint32_t   gIsPrepassEnabled                         = 0;
	} root_shader_parameter;

	std::shared_ptr<RHIDescriptor>    descriptor     = std::move(renderer->GetRHIContext()->CreateDescriptor(meta, true));
	std::shared_ptr<RHIPipelineState> pipeline_state = std::move(renderer->GetRHIContext()->CreatePipelineState(true));

	std::shared_ptr<RHIBuffer> root_shader_parameter_buffer = std::move(renderer->GetRHIContext()->CreateBuffer<decltype(root_shader_parameter)>(1, RHIBufferUsage::ConstantBuffer, RHIMemoryUsage::CPU_TO_GPU, true));
	root_shader_parameter_buffer->CopyToDevice(&root_shader_parameter, sizeof(root_shader_parameter), 0);

	std::shared_ptr<RHIBuffer> constant_buffer = std::move(renderer->GetRHIContext()->CreateBuffer<glm::mat4>(1, RHIBufferUsage::ConstantBuffer, RHIMemoryUsage::CPU_TO_GPU, true));
	glm::mat4                  identity        = glm::mat4(1);
	constant_buffer->CopyToDevice(&identity, sizeof(identity), 0);

	pipeline_state->SetShader(RHIShaderStage::Compute, shader);

	renderer->GetResourceManager()->ImportTexture("./Asset/NRD_Data/Sample/gIn_Normal_Roughness.dds", true, false);
	renderer->GetResourceManager()->ImportTexture("./Asset/NRD_Data/Sample/gIn_ViewZ.dds", true, false);
	renderer->GetResourceManager()->ImportTexture("./Asset/NRD_Data/Sample/gIn_ObjectMotion.dds", true, false);
	renderer->GetResourceManager()->ImportTexture("./Asset/NRD_Data/Sample/gIn_Prev_ViewZ.dds", true, false);
	renderer->GetResourceManager()->ImportTexture("./Asset/NRD_Data/Sample/gIn_Prev_Normal_Roughness.dds", true, false);
	renderer->GetResourceManager()->ImportTexture("./Asset/NRD_Data/Sample/gIn_Prev_AccumSpeeds_MaterialID.dds", true, false);
	renderer->GetResourceManager()->ImportTexture("./Asset/NRD_Data/Sample/gIn_Diff.dds", true, false);
	renderer->GetResourceManager()->ImportTexture("./Asset/NRD_Data/Sample/gIn_Diff_History.dds", true, false);

	auto *orig_In_Normal_Roughness            = renderer->GetResourceManager()->GetTexture(std::to_string(Hash(std::string("./Asset/NRD_Data/Sample/gIn_Normal_Roughness.dds"), true)))->texture.get();
	auto *orig_In_ViewZ                       = renderer->GetResourceManager()->GetTexture(std::to_string(Hash(std::string("./Asset/NRD_Data/Sample/gIn_ViewZ.dds"), true)))->texture.get();
	auto *orig_In_ObjectMotion                = renderer->GetResourceManager()->GetTexture(std::to_string(Hash(std::string("./Asset/NRD_Data/Sample/gIn_ObjectMotion.dds"), true)))->texture.get();
	auto *orig_In_Prev_ViewZ                  = renderer->GetResourceManager()->GetTexture(std::to_string(Hash(std::string("./Asset/NRD_Data/Sample/gIn_Prev_ViewZ.dds"), true)))->texture.get();
	auto *orig_In_Prev_Normal_Roughness       = renderer->GetResourceManager()->GetTexture(std::to_string(Hash(std::string("./Asset/NRD_Data/Sample/gIn_Prev_Normal_Roughness.dds"), true)))->texture.get();
	auto *orig_In_Prev_AccumSpeeds_MaterialID = renderer->GetResourceManager()->GetTexture(std::to_string(Hash(std::string("./Asset/NRD_Data/Sample/gIn_Prev_AccumSpeeds_MaterialID.dds"), true)))->texture.get();
	auto *orig_In_Diff                        = renderer->GetResourceManager()->GetTexture(std::to_string(Hash(std::string("./Asset/NRD_Data/Sample/gIn_Diff.dds"), true)))->texture.get();
	auto *orig_In_Diff_History                = renderer->GetResourceManager()->GetTexture(std::to_string(Hash(std::string("./Asset/NRD_Data/Sample/gIn_Diff_History.dds"), true)))->texture.get();

	// Convert to Float32 format
	std::shared_ptr<RHITexture> gfx_In_Normal_Roughness            = std::move(renderer->GetRHIContext()->CreateTexture2D(orig_In_Normal_Roughness->GetDesc().width, orig_In_Normal_Roughness->GetDesc().height, RHIFormat::R32G32B32A32_FLOAT, RHITextureUsage::Transfer, false, 1, true));
	std::shared_ptr<RHITexture> gfx_In_ViewZ                       = std::move(renderer->GetRHIContext()->CreateTexture2D(orig_In_ViewZ->GetDesc().width, orig_In_ViewZ->GetDesc().height, RHIFormat::R32_FLOAT, RHITextureUsage::Transfer, false, 1, true));
	std::shared_ptr<RHITexture> gfx_In_ObjectMotion                = std::move(renderer->GetRHIContext()->CreateTexture2D(orig_In_ObjectMotion->GetDesc().width, orig_In_ObjectMotion->GetDesc().height, RHIFormat::R32G32B32A32_FLOAT, RHITextureUsage::Transfer, false, 1, true));
	std::shared_ptr<RHITexture> gfx_In_Prev_ViewZ                  = std::move(renderer->GetRHIContext()->CreateTexture2D(orig_In_Prev_ViewZ->GetDesc().width, orig_In_Prev_ViewZ->GetDesc().height, RHIFormat::R32_FLOAT, RHITextureUsage::Transfer, false, 1, true));
	std::shared_ptr<RHITexture> gfx_In_Prev_Normal_Roughness       = std::move(renderer->GetRHIContext()->CreateTexture2D(orig_In_Prev_Normal_Roughness->GetDesc().width, orig_In_Prev_Normal_Roughness->GetDesc().height, RHIFormat::R32G32B32A32_FLOAT, RHITextureUsage::Transfer, false, 1, true));
	std::shared_ptr<RHITexture> gfx_In_Prev_AccumSpeeds_MaterialID = std::move(renderer->GetRHIContext()->CreateTexture2D(orig_In_Prev_AccumSpeeds_MaterialID->GetDesc().width, orig_In_Prev_AccumSpeeds_MaterialID->GetDesc().height, RHIFormat::R32_FLOAT, RHITextureUsage::Transfer, false, 1, true));
	std::shared_ptr<RHITexture> gfx_In_Diff                        = std::move(renderer->GetRHIContext()->CreateTexture2D(orig_In_Diff->GetDesc().width, orig_In_Diff->GetDesc().height, RHIFormat::R32_FLOAT, RHITextureUsage::Transfer, false, 1, true));
	std::shared_ptr<RHITexture> gfx_In_Diff_History                = std::move(renderer->GetRHIContext()->CreateTexture2D(orig_In_Diff_History->GetDesc().width, orig_In_Diff_History->GetDesc().height, RHIFormat::R32_FLOAT, RHITextureUsage::Transfer, false, 1, true));

	{
		auto *cmd_buffer = renderer->GetRHIContext()->CreateCommand(RHIQueueFamily::Graphics);
		cmd_buffer->Begin();
		cmd_buffer->BlitTexture(orig_In_Normal_Roughness, TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}, RHIResourceState::ShaderResource, gfx_In_Normal_Roughness.get(), TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}, RHIResourceState::Undefined, RHIFilter::Nearest);
		cmd_buffer->BlitTexture(orig_In_ViewZ, TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}, RHIResourceState::ShaderResource, gfx_In_ViewZ.get(), TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}, RHIResourceState::Undefined, RHIFilter::Nearest);
		cmd_buffer->BlitTexture(orig_In_ObjectMotion, TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}, RHIResourceState::ShaderResource, gfx_In_ObjectMotion.get(), TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}, RHIResourceState::Undefined, RHIFilter::Nearest);
		cmd_buffer->BlitTexture(orig_In_Prev_ViewZ, TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}, RHIResourceState::ShaderResource, gfx_In_Prev_ViewZ.get(), TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}, RHIResourceState::Undefined, RHIFilter::Nearest);
		cmd_buffer->BlitTexture(orig_In_Prev_Normal_Roughness, TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}, RHIResourceState::ShaderResource, gfx_In_Prev_Normal_Roughness.get(), TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}, RHIResourceState::Undefined, RHIFilter::Nearest);
		cmd_buffer->BlitTexture(orig_In_Prev_AccumSpeeds_MaterialID, TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}, RHIResourceState::ShaderResource, gfx_In_Prev_AccumSpeeds_MaterialID.get(), TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}, RHIResourceState::Undefined, RHIFilter::Nearest);
		cmd_buffer->BlitTexture(orig_In_Diff, TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}, RHIResourceState::ShaderResource, gfx_In_Diff.get(), TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}, RHIResourceState::Undefined, RHIFilter::Nearest);
		cmd_buffer->BlitTexture(orig_In_Diff_History, TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}, RHIResourceState::ShaderResource, gfx_In_Diff_History.get(), TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}, RHIResourceState::Undefined, RHIFilter::Nearest);
		cmd_buffer->End();
		renderer->GetRHIContext()->Execute(cmd_buffer);
	}

	std::shared_ptr<RHITexture> In_Normal_Roughness            = renderer->GetRHIContext()->MapToCUDATexture(gfx_In_Normal_Roughness.get());
	std::shared_ptr<RHITexture> In_ViewZ                       = renderer->GetRHIContext()->MapToCUDATexture(gfx_In_ViewZ.get());
	std::shared_ptr<RHITexture> In_ObjectMotion                = renderer->GetRHIContext()->MapToCUDATexture(gfx_In_ObjectMotion.get());
	std::shared_ptr<RHITexture> In_Prev_ViewZ                  = renderer->GetRHIContext()->MapToCUDATexture(gfx_In_Prev_ViewZ.get());
	std::shared_ptr<RHITexture> In_Prev_Normal_Roughness       = renderer->GetRHIContext()->MapToCUDATexture(gfx_In_Prev_Normal_Roughness.get());
	std::shared_ptr<RHITexture> In_Prev_AccumSpeeds_MaterialID = renderer->GetRHIContext()->MapToCUDATexture(gfx_In_Prev_AccumSpeeds_MaterialID.get());
	std::shared_ptr<RHITexture> In_Diff                        = renderer->GetRHIContext()->MapToCUDATexture(gfx_In_Diff.get());
	std::shared_ptr<RHITexture> In_Diff_History                = renderer->GetRHIContext()->MapToCUDATexture(gfx_In_Diff_History.get());

	return [=](RenderGraph &render_graph, RHICommand *cmd_buffer, rttr::variant &config) {
		auto *out_diff = render_graph.GetCUDATexture(desc.resources.at("Out_Diff").handle);
		auto *out_data = render_graph.GetCUDATexture(desc.resources.at("Out_Data1").handle);

		descriptor
		    ->BindBuffer("_RootShaderParameters", root_shader_parameter_buffer.get())
		    .BindSampler("gLinearClamp", renderer->GetRHIContext()->CreateSampler(SamplerDesc::LinearClamp))
		    .BindTexture("gIn_Normal_Roughness", In_Normal_Roughness.get(), RHITextureDimension::Texture2D)
		    .BindTexture("gIn_ViewZ", In_ViewZ.get(), RHITextureDimension::Texture2D)
		    .BindTexture("gIn_ObjectMotion", In_ObjectMotion.get(), RHITextureDimension::Texture2D)
		    .BindTexture("gIn_Prev_ViewZ", In_Prev_ViewZ.get(), RHITextureDimension::Texture2D)
		    .BindTexture("gIn_Prev_Normal_Roughness", In_Prev_Normal_Roughness.get(), RHITextureDimension::Texture2D)
		    .BindTexture("gIn_Prev_AccumSpeeds_MaterialID", In_Prev_AccumSpeeds_MaterialID.get(), RHITextureDimension::Texture2D)
		    .BindTexture("gIn_Diff", In_Diff.get(), RHITextureDimension::Texture2D)
		    .BindTexture("gIn_Diff_History", In_Diff_History.get(), RHITextureDimension::Texture2D)
		    .BindTexture("gOut_Diff", out_diff, RHITextureDimension::Texture2D)
		    .BindTexture("gOut_Data1", out_data, RHITextureDimension::Texture2D);

		cmd_buffer->BindDescriptor(descriptor.get());
		cmd_buffer->BindPipelineState(pipeline_state.get());
		cmd_buffer->Dispatch(out_diff->GetDesc().width, out_diff->GetDesc().height, 1, 8, 8, 1);
	};
}
}        // namespace Ilum