#include "IPass.hpp"

#include <Material/MaterialData.hpp>
#include <Resource/Resource/Mesh.hpp>
#include <Resource/Resource/SkinnedMesh.hpp>
#include <Resource/ResourceManager.hpp>
#include <Scene/Components/AllComponents.hpp>
#include <Scene/Scene.hpp>

using namespace Ilum;

class RayTracedAO : public RenderPass<RayTracedAO>
{
	struct Config
	{
		uint32_t max_bounce      = 5;
		uint32_t max_spp         = 100;
		uint32_t frame_count     = 0;
		float    clamp_threshold = 4.f;
		uint32_t anti_aliasing   = 1;
	};

	struct PipelineDesc
	{
		std::shared_ptr<RHIPipelineState> pipeline_state = nullptr;
		ShaderMeta                        shader_meta;
	};

	struct RayTracedPassData
	{
		std::unique_ptr<RHITexture>                raytrace = nullptr;
		std::array<std::unique_ptr<RHITexture>, 2> ao{nullptr};
		std::array<std::unique_ptr<RHITexture>, 2> history_length{nullptr};
		std::array<std::unique_ptr<RHITexture>, 2> bilateral_blur{nullptr};
	};

  public:
	RayTracedAO() = default;

	~RayTracedAO() = default;

	virtual RenderPassDesc Create(size_t &handle)
	{
		RenderPassDesc desc;
		return desc.SetBindPoint(BindPoint::RayTracing)
		    .SetName("RayTracedAO")
		    .SetCategory("AO")
		    .SetConfig(Config())
		    .WriteTexture2D(handle++, "Output", RHIFormat::R32G32B32A32_FLOAT, RHIResourceState::UnorderedAccess);
	}

	virtual void CreateCallback(RenderGraph::RenderTask *task, const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer)
	{
		struct
		{
			std::shared_ptr<RHIPipelineState> pipeline = nullptr;
		} pipeline;

		pipeline.pipeline = std::shared_ptr<RHIPipelineState>(std::move(renderer->GetRHIContext()->CreatePipelineState()));

		std::shared_ptr<RHIBuffer> config_buffer = std::shared_ptr<RHIBuffer>(std::move(renderer->GetRHIContext()->CreateBuffer(sizeof(Config), RHIBufferUsage::ConstantBuffer, RHIMemoryUsage::CPU_TO_GPU)));

		*task = [=](RenderGraph &render_graph, RHICommand *cmd_buffer, Variant &config, RenderGraphBlackboard &black_board) {
			auto  output      = render_graph.GetTexture(desc.GetPin("Output").handle);
			auto *gpu_scene   = black_board.Get<GPUScene>();
			auto *view        = black_board.Get<View>();
			auto *rhi_context = renderer->GetRHIContext();
			auto *config_data = config.Convert<Config>();

			config_buffer->CopyToDevice(config_data, sizeof(Config));
		};
	}

	virtual void OnImGui(Variant *config)
	{
	}
};

CONFIGURATION_PASS(RayTracedAO)