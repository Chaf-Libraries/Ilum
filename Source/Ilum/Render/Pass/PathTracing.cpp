#include "PathTracing.hpp"

#include <RHI/DescriptorState.hpp>
#include <RHI/FrameBuffer.hpp>

#include <Render/RGBuilder.hpp>
#include <Render/Renderer.hpp>

#include <Scene/Scene.hpp>
#include <Scene/Entity.hpp>
#include <Scene/Component/Camera.hpp>

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

	auto visibility_buffer = builder.CreateTexture(
	    "Visibility Buffer",
	    TextureDesc{
	        builder.GetRenderer().GetExtent().width,  /*width*/
	        builder.GetRenderer().GetExtent().height, /*height*/
	        1,                                        /*depth*/
	        1,                                        /*mips*/
	        1,                                        /*layers*/
	        VK_SAMPLE_COUNT_1_BIT,
	        VK_FORMAT_R32_UINT,
	        VK_IMAGE_USAGE_SAMPLED_BIT},
	    TextureState{VK_IMAGE_USAGE_SAMPLED_BIT});

	pass->AddResource(shading);
	pass->AddResource(visibility_buffer);

	TextureViewDesc view_desc  = {};
	view_desc.aspect           = VK_IMAGE_ASPECT_COLOR_BIT;
	view_desc.view_type        = VK_IMAGE_VIEW_TYPE_2D;
	view_desc.base_array_layer = 0;
	view_desc.base_mip_level   = 0;
	view_desc.layer_count      = 1;
	view_desc.level_count      = 1;

	ShaderDesc raygen_shader  = {};
	raygen_shader.filename    = "./Source/Shaders/PathTracing.hlsl";
	raygen_shader.entry_point = "RayGen";
	raygen_shader.stage       = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
	raygen_shader.type        = ShaderType::HLSL;

	PipelineState pso;
	pso
	    .SetName("PathTracing")
	    .LoadShader(raygen_shader);

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
		if (!camera_buffer)
		{
			return;
		}


		cmd_buffer.Bind(pso);
		cmd_buffer.Bind(
		    cmd_buffer.GetDescriptorState()
		        .Bind(0, 0, resource.GetTexture(shading)->GetView(view_desc))
		        //.Bind(0, 1, &renderer.GetScene()->GetTLAS())
		        .Bind(0, 2, camera_buffer)
		        .Bind(0, 3, resource.GetTexture(visibility_buffer)->GetView(view_desc))
		        .Bind(0, 4, renderer.GetScene()->GetInstanceBuffer())
		        .Bind(0, 5, renderer.GetScene()->GetAssetManager().GetMeshletBuffer())
		        .Bind(0, 6, renderer.GetScene()->GetAssetManager().GetVertexBuffer())
		        .Bind(0, 7, renderer.GetScene()->GetAssetManager().GetMeshletVertexBuffer())
		        .Bind(0, 8, renderer.GetScene()->GetAssetManager().GetMeshletTriangleBuffer())
		        .Bind(0, 9, renderer.GetScene()->GetAssetManager().GetMaterialBuffer())

		        //.Bind(0, 5, directional_lights)
		        //.Bind(0, 6, spot_lights)
		        //.Bind(0, 7, point_lights)
		        //.Bind(0, 8, area_lights)
		        //.Bind(0, 9, shadowmaps)
		        //.Bind(0, 10, cascade_shadowmaps)
		        //.Bind(0, 11, onmishadowmaps)
		        .Bind(0, 12, renderer.GetScene()->GetAssetManager().GetTextureViews())
		        .Bind(0, 13, renderer.GetSampler(SamplerType::TrilinearWarp)));

		cmd_buffer.TraceRays(renderer.GetExtent().width, renderer.GetExtent().height);
	});

	pass->BindImGui([=](ImGuiContext &, const RGResources &) {

	});

	builder.AddPass(std::move(pass));
}
}        // namespace Ilum