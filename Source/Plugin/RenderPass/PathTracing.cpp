#include "IPass.hpp"

#include <Material/MaterialData.hpp>
#include <Resource/Resource/Mesh.hpp>
#include <Resource/Resource/SkinnedMesh.hpp>
#include <Resource/ResourceManager.hpp>
#include <Scene/Components/AllComponents.hpp>
#include <Scene/Scene.hpp>

using namespace Ilum;

class PathTracing : public RenderPass<PathTracing>
{
	struct Config
	{
		uint32_t max_bounce      = 5;
		uint32_t max_spp         = 100;
		uint32_t frame_count     = 0;
		float    clamp_threshold = 4.f;
		uint32_t     anti_aliasing   = 1;
	};

  public:
	PathTracing() = default;

	~PathTracing() = default;

	virtual RenderPassDesc Create(size_t &handle)
	{
		RenderPassDesc desc;
		return desc.SetBindPoint(BindPoint::RayTracing)
		    .SetName("PathTracing")
		    .SetCategory("RayTracing")
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

			if (renderer->GetScene()->IsUpdate())
			{
				config_data->frame_count = 0;
			}
			else
			{
				config_data->frame_count = glm::clamp(config_data->frame_count, 0u, config_data->max_spp);
			}

			if (config_data->frame_count == config_data->max_spp)
			{
				return;
			}

			// Setup material pipeline
			ShaderMeta meta;
			{
				pipeline.pipeline->ClearShader();

				auto *raygen_shader     = renderer->RequireShader("Source/Shaders/PathTracing.hlsl", "RayGenMain", RHIShaderStage::RayGen, {"RAYGEN_SHADER", "RAYTRACING_PIPELINE", gpu_scene->textures.texture_cube ? "USE_SKYBOX" : "NO_SKYBOX"});
				auto *closesthit_shader = renderer->RequireShader("Source/Shaders/PathTracing.hlsl", "ClosesthitMain", RHIShaderStage::ClosestHit, {"CLOSESTHIT_SHADER", "RAYTRACING_PIPELINE"}, {"Material/Material.hlsli"});
				auto *miss_shader       = renderer->RequireShader("Source/Shaders/PathTracing.hlsl", "MissMain", RHIShaderStage::Miss, {"MISS_SHADER", "RAYTRACING_PIPELINE"});

				meta += renderer->RequireShaderMeta(raygen_shader);
				meta += renderer->RequireShaderMeta(closesthit_shader);
				meta += renderer->RequireShaderMeta(miss_shader);

				pipeline.pipeline->SetShader(RHIShaderStage::RayGen, raygen_shader);
				pipeline.pipeline->SetShader(RHIShaderStage::ClosestHit, closesthit_shader);
				pipeline.pipeline->SetShader(RHIShaderStage::Miss, miss_shader);

				for (const auto &data : gpu_scene->material.data)
				{
					auto *material_closesthit_shader = renderer->RequireShader("Source/Shaders/PathTracing.hlsl", "ClosesthitMain", RHIShaderStage::ClosestHit, {"CLOSESTHIT_SHADER", "RAYTRACING_PIPELINE", data->signature}, {data->shader});
					pipeline.pipeline->SetShader(RHIShaderStage::ClosestHit, material_closesthit_shader);
					meta += renderer->RequireShaderMeta(material_closesthit_shader);
				}
			}

			if (gpu_scene->mesh_buffer.instance_count > 0)
			{
				auto *descriptor = rhi_context->CreateDescriptor(meta);
				descriptor->BindAccelerationStructure("TopLevelAS", gpu_scene->TLAS.get())
				    .BindTexture("Output", output, RHITextureDimension::Texture2D)
				    .BindBuffer("ConfigBuffer", config_buffer.get())
				    .BindBuffer("InstanceBuffer", gpu_scene->mesh_buffer.instances.get())
				    .BindBuffer("ViewBuffer", view->buffer.get())
				    .BindBuffer("VertexBuffer", gpu_scene->mesh_buffer.vertex_buffers)
				    .BindBuffer("IndexBuffer", gpu_scene->mesh_buffer.index_buffers)
				    .BindTexture("Textures", gpu_scene->textures.texture_2d, RHITextureDimension::Texture2D)
				    .BindSampler("Samplers", gpu_scene->samplers)
				    .BindBuffer("MaterialOffsets", gpu_scene->material.material_offset.get())
				    .BindBuffer("MaterialBuffer", gpu_scene->material.material_buffer.get())
				    .BindBuffer("PointLightBuffer", gpu_scene->light.point_light_buffer.get())
				    .BindBuffer("SpotLightBuffer", gpu_scene->light.spot_light_buffer.get())
				    .BindBuffer("DirectionalLightBuffer", gpu_scene->light.directional_light_buffer.get())
				    .BindBuffer("RectLightBuffer", gpu_scene->light.rect_light_buffer.get())
				    .BindBuffer("LightInfoBuffer", gpu_scene->light.light_info_buffer.get());

				if (gpu_scene->textures.texture_cube)
				{
					descriptor->BindTexture("Skybox", gpu_scene->textures.texture_cube, RHITextureDimension::TextureCube)
					    .BindSampler("SkyboxSampler", rhi_context->CreateSampler(SamplerDesc::LinearClamp()));
				}

				cmd_buffer->BindDescriptor(descriptor);
				cmd_buffer->BindPipelineState(pipeline.pipeline.get());
				cmd_buffer->TraceRay(output->GetDesc().width, output->GetDesc().height, 1);

				config_data->frame_count++;
			}
			else
			{
				config_data->frame_count = 0;
			}
		};
	}

	virtual void OnImGui(Variant *config)
	{
		auto *config_data = config->Convert<Config>();

		ImGui::Text("SPP: %d", config_data->frame_count);

		 if (ImGui::Checkbox("Anti-Aliasing", reinterpret_cast<bool *>(&config_data->anti_aliasing)))
		{
			config_data->frame_count = 0;
		 }

		if (ImGui::SliderInt("Max Bounce", reinterpret_cast<int32_t *>(&config_data->max_bounce), 1, 100))
		{
			config_data->frame_count = 0;
		}

		if (ImGui::DragInt("Max SPP", reinterpret_cast<int32_t *>(&config_data->max_spp), 0.1f, 1, std::numeric_limits<int32_t>::max()))
		{
			config_data->frame_count = 0;
		}

		ImGui::ProgressBar(static_cast<float>(config_data->frame_count) / static_cast<float>(config_data->max_spp),
		                   ImVec2(0.f, 0.f),
		                   (std::to_string(config_data->frame_count) + "/" + std::to_string(config_data->max_spp)).c_str());

		if (ImGui::DragFloat("Clamp Threshold", &config_data->clamp_threshold, 0.1f, 0.0f, std::numeric_limits<float>::max()))
		{
			config_data->frame_count = 0;
		}
	}
};

CONFIGURATION_PASS(PathTracing)