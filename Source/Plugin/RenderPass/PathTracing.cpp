#include "IPass.hpp"

#include <Resource/Resource/Mesh.hpp>
#include <Resource/Resource/SkinnedMesh.hpp>
#include <Resource/ResourceManager.hpp>
#include <Scene/Components/AllComponents.hpp>
#include <Scene/Scene.hpp>

using namespace Ilum;

class PathTracing : public IPass<PathTracing>
{
  public:
	PathTracing() = default;

	~PathTracing() = default;

	virtual void CreateDesc(RenderPassDesc *desc)
	{
		desc->SetBindPoint(BindPoint::RayTracing)
		    .Write("Output", RenderResourceDesc::Type::Texture, RHIResourceState::UnorderedAccess);
	}

	virtual void CreateCallback(RenderGraph::RenderTask *task, const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer)
	{
		struct
		{
			std::shared_ptr<RHIPipelineState> pipeline = nullptr;

			ShaderMeta meta;
		} pipeline;

		pipeline.pipeline = std::shared_ptr<RHIPipelineState>(std::move(renderer->GetRHIContext()->CreatePipelineState()));

		auto *raygen_shader     = renderer->RequireShader("Source/Shaders/PathTracing.hlsl", "RayGenMain", RHIShaderStage::RayGen, {"RAYGEN_SHADER", "RAYTRACING_PIPELINE"});
		auto *closesthit_shader = renderer->RequireShader("Source/Shaders/PathTracing.hlsl", "ClosesthitMain", RHIShaderStage::ClosestHit, {"CLOSESTHIT_SHADER", "RAYTRACING_PIPELINE"});
		auto *miss_shader       = renderer->RequireShader("Source/Shaders/PathTracing.hlsl", "MissMain", RHIShaderStage::Miss, {"MISS_SHADER", "RAYTRACING_PIPELINE"});

		pipeline.meta = renderer->RequireShaderMeta(raygen_shader);
		pipeline.meta += renderer->RequireShaderMeta(closesthit_shader);
		pipeline.meta += renderer->RequireShaderMeta(miss_shader);

		pipeline.pipeline->SetShader(RHIShaderStage::RayGen, raygen_shader);
		pipeline.pipeline->SetShader(RHIShaderStage::ClosestHit, closesthit_shader);
		pipeline.pipeline->SetShader(RHIShaderStage::Miss, miss_shader);

		*task = [=](RenderGraph &render_graph, RHICommand *cmd_buffer, Variant &config, RenderGraphBlackboard &black_board) {
			auto  output      = render_graph.GetTexture(desc.resources.at("Output").handle);
			auto *gpu_scene   = black_board.Get<GPUScene>();
			auto *view        = black_board.Get<View>();
			auto *rhi_context = renderer->GetRHIContext();

			if (gpu_scene->mesh_buffer.instance_count > 0)
			{
				auto *descriptor = rhi_context->CreateDescriptor(pipeline.meta);
				descriptor->BindAccelerationStructure("TopLevelAS", gpu_scene->TLAS.get())
				    .BindTexture("Output", output, RHITextureDimension::Texture2D)
				    .BindBuffer("ViewBuffer", view->buffer.get());
				auto *cmd_buffer_ = rhi_context->CreateCommand(RHIQueueFamily::Compute);
				cmd_buffer->BindDescriptor(descriptor);
				cmd_buffer->BindPipelineState(pipeline.pipeline.get());
				cmd_buffer->TraceRay(output->GetDesc().width, output->GetDesc().height, 1);
			}
		};
	}

	virtual void OnImGui(Variant *config)
	{
	}
};

CONFIGURATION_PASS(PathTracing)