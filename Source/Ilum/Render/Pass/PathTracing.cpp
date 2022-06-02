#include "PathTracing.hpp"

#include <RHI/DescriptorState.hpp>
#include <RHI/FrameBuffer.hpp>

#include <Render/RGBuilder.hpp>
#include <Render/Renderer.hpp>

#include <Scene/Component/Camera.hpp>
#include <Scene/Component/Light.hpp>
#include <Scene/Entity.hpp>
#include <Scene/Scene.hpp>

#include <Asset/AssetManager.hpp>

namespace Ilum
{
PathTracing::PathTracing() :
    RenderPass("PathTracing")
{
}

void PathTracing::Create(RGBuilder &builder)
{
	std::unique_ptr<RenderPass> pass = std::make_unique<PathTracing>();

	// Render Target
	auto shading = builder.CreateTexture(
	    "Shading",
	    TextureDesc{
	        builder.GetRenderer().GetExtent().width,  /*width*/
	        builder.GetRenderer().GetExtent().height, /*height*/
	        1,                                        /*depth*/
	        1,                                        /*mips*/
	        1,                                        /*layers*/
	        VK_SAMPLE_COUNT_1_BIT,
	        VK_FORMAT_R32G32B32A32_SFLOAT,
	        VK_IMAGE_USAGE_STORAGE_BIT},
	    TextureState{VK_IMAGE_USAGE_STORAGE_BIT});

	pass->AddResource(shading);

	TextureViewDesc view_desc  = {};
	view_desc.aspect           = VK_IMAGE_ASPECT_COLOR_BIT;
	view_desc.view_type        = VK_IMAGE_VIEW_TYPE_2D;
	view_desc.base_array_layer = 0;
	view_desc.base_mip_level   = 0;
	view_desc.layer_count      = 1;
	view_desc.level_count      = 1;

	ShaderDesc pathtracing_compute_shader  = {};
	pathtracing_compute_shader.filename    = "./Source/Shaders/PathTracing/PathTracingCompute.hlsl";
	pathtracing_compute_shader.entry_point = "main";
	pathtracing_compute_shader.stage       = VK_SHADER_STAGE_COMPUTE_BIT;
	pathtracing_compute_shader.type        = ShaderType::HLSL;

	PipelineState pathtracing_compute_pso;
	pathtracing_compute_pso
	    .SetName("PathTracing - Compute")
	    .LoadShader(pathtracing_compute_shader);

	pass->BindCallback([=](CommandBuffer &cmd_buffer, const RGResources &resource, Renderer &renderer) {
		if (renderer.GetScene()->GetInstanceBuffer().empty())
		{
			return;
		}

		Entity camera_entity = Entity(*renderer.GetScene(), renderer.GetScene()->GetMainCamera());
		if (!camera_entity.IsValid())
		{
			return;
		}

		auto *camera_buffer = camera_entity.GetComponent<cmpt::Camera>().GetBuffer();

		struct
		{
			uint32_t directional_light_count = 0;
			uint32_t spot_light_count        = 0;
			uint32_t point_light_count       = 0;
			uint32_t area_light_count        = 0;
		} push_constants;

		std::vector<Buffer *> directional_lights;
		std::vector<Buffer *> spot_lights;
		std::vector<Buffer *> point_lights;
		std::vector<Buffer *> area_lights;

		{
			auto view = renderer.GetScene()->GetRegistry().view<cmpt::Light>();

			directional_lights.reserve(view.size());
			spot_lights.reserve(view.size());
			point_lights.reserve(view.size());
			area_lights.reserve(view.size());

			view.each([&](cmpt::Light &light) {
				switch (light.GetType())
				{
					case cmpt::LightType::Point:
						point_lights.push_back(light.GetBuffer());
						break;
					case cmpt::LightType::Directional:
						directional_lights.push_back(light.GetBuffer());
						break;
					case cmpt::LightType::Spot:
						spot_lights.push_back(light.GetBuffer());
						break;
					case cmpt::LightType::Area:
						area_lights.push_back(light.GetBuffer());
						break;
					default:
						break;
				}
			});

			push_constants.directional_light_count = static_cast<uint32_t>(directional_lights.size());
			push_constants.spot_light_count        = static_cast<uint32_t>(spot_lights.size());
			push_constants.point_light_count       = static_cast<uint32_t>(point_lights.size());
			push_constants.area_light_count        = static_cast<uint32_t>(area_lights.size());
		}

		if (renderer.GetDevice().IsRayTracingEnable())
		{
		}
		else
		{
			auto view = renderer.GetScene()->GetRegistry().view<cmpt::MeshRenderer>();

			if (view.empty())
			{
				return;
			}

			std::vector<Buffer *> blas_buffer;
			blas_buffer.reserve(view.size());

			view.each([&](cmpt::MeshRenderer &mesh_renderer) {
				auto *mesh = mesh_renderer.GetMesh();
				if (mesh)
				{
					blas_buffer.push_back(&mesh->GetBLAS().GetBVHBuffer());
				}
			});

			cmd_buffer.Bind(pathtracing_compute_pso);
			cmd_buffer.Bind(
			    cmd_buffer.GetDescriptorState()
			        .Bind(0, 0, resource.GetTexture(shading)->GetView(view_desc))
			        .Bind(0, 1, camera_buffer)
			        .Bind(0, 2, &renderer.GetScene()->GetTLAS().GetBVHBuffer())
			        .Bind(0, 3, blas_buffer)
			        .Bind(1, 0, renderer.GetScene()->GetInstanceBuffer())
			        .Bind(1, 1, renderer.GetScene()->GetAssetManager().GetMeshletBuffer())
			        .Bind(1, 2, renderer.GetScene()->GetAssetManager().GetVertexBuffer())
			        .Bind(1, 3, renderer.GetScene()->GetAssetManager().GetMeshletVertexBuffer())
			        .Bind(1, 4, renderer.GetScene()->GetAssetManager().GetMeshletTriangleBuffer())
			        .Bind(1, 5, renderer.GetScene()->GetAssetManager().GetIndexBuffer())
			        .Bind(2, 0, renderer.GetScene()->GetAssetManager().GetMaterialBuffer())
			        .Bind(2, 1, renderer.GetScene()->GetAssetManager().GetTextureViews())
			        .Bind(2, 2, renderer.GetSampler(SamplerType::TrilinearWarp))
			        .Bind(3, 0, directional_lights)
			        .Bind(3, 1, spot_lights)
			        .Bind(3, 2, point_lights)
			        .Bind(3, 3, area_lights));

			cmd_buffer.PushConstants(VK_SHADER_STAGE_COMPUTE_BIT, &push_constants, sizeof(push_constants), 0);
			cmd_buffer.Dispatch((renderer.GetExtent().width + 32 - 1) / 32, (renderer.GetExtent().height + 32 - 1) / 32);
		}

		// cmd_buffer.TraceRays(renderer.GetExtent().width, renderer.GetExtent().height);
	});

	pass->BindImGui([=](ImGuiContext &, const RGResources &) {

	});

	builder.AddPass(std::move(pass));
}
}        // namespace Ilum