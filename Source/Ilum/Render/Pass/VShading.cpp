#include "VShading.hpp"

#include <RHI/DescriptorState.hpp>
#include <RHI/FrameBuffer.hpp>

#include <Render/RGBuilder.hpp>
#include <Render/Renderer.hpp>

#include <Scene/Component/Camera.hpp>
#include <Scene/Scene.hpp>
#include <Scene/Entity.hpp>

#include <Asset/AssetManager.hpp>

namespace Ilum
{
VShading::VShading() :
    RenderPass("VShading")
{
}

void VShading::Create(RGBuilder &builder)
{
	std::unique_ptr<RenderPass> pass = std::make_unique<VShading>();

	// Render Target
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

	auto shading = builder.CreateTexture(
	    "Shading",
	    TextureDesc{
	        builder.GetRenderer().GetExtent().width,  /*width*/
	        builder.GetRenderer().GetExtent().height, /*height*/
	        1,                                        /*depth*/
	        1,                                        /*mips*/
	        1,                                        /*layers*/
	        VK_SAMPLE_COUNT_1_BIT,
	        VK_FORMAT_R16G16B16A16_SFLOAT,
	        VK_IMAGE_USAGE_STORAGE_BIT},
	    TextureState{VK_IMAGE_USAGE_STORAGE_BIT});

	auto normal = builder.CreateTexture(
	    "Normal",
	    TextureDesc{
	        builder.GetRenderer().GetExtent().width,  /*width*/
	        builder.GetRenderer().GetExtent().height, /*height*/
	        1,                                        /*depth*/
	        1,                                        /*mips*/
	        1,                                        /*layers*/
	        VK_SAMPLE_COUNT_1_BIT,
	        VK_FORMAT_R16G16_SFLOAT,
	        VK_IMAGE_USAGE_STORAGE_BIT},
	    TextureState{VK_IMAGE_USAGE_STORAGE_BIT});

	pass->AddResource(visibility_buffer);
	pass->AddResource(shading);
	pass->AddResource(normal);

	TextureViewDesc view_desc  = {};
	view_desc.aspect           = VK_IMAGE_ASPECT_COLOR_BIT;
	view_desc.view_type        = VK_IMAGE_VIEW_TYPE_2D;
	view_desc.base_array_layer = 0;
	view_desc.base_mip_level   = 0;
	view_desc.layer_count      = 1;
	view_desc.level_count      = 1;

	ShaderDesc shader  = {};
	shader.filename    = "./Source/Shaders/VShading.hlsl";
	shader.entry_point = "main";
	shader.stage       = VK_SHADER_STAGE_COMPUTE_BIT;
	shader.type        = ShaderType::HLSL;

	PipelineState pso;
	pso
	    .SetName("VShading")
	    .LoadShader(shader);

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

		std::vector<Buffer *> directional_lights;
		std::vector<Buffer *> spot_lights;
		std::vector<Buffer *> point_lights;
		std::vector<Buffer *> area_lights;

		std::vector<VkImageView> shadowmaps;
		std::vector<VkImageView> cascade_shadowmaps;
		std::vector<VkImageView> onmishadowmaps;

		/*auto view = renderer.GetScene()->GetRegistry().view<cmpt::Light>();
		view.each([&](cmpt::Light &light) {
			switch (light.type)
			{
				case cmpt::LightType::Point:
					point_lights.push_back(light.buffer.get());
					onmishadowmaps.push_back(light.shadow_map->GetView(TextureViewDesc{
					    VK_IMAGE_VIEW_TYPE_CUBE,
					    VK_IMAGE_ASPECT_DEPTH_BIT,
					    0,
					    1,
					    0,
					    6}));
					break;
				case cmpt::LightType::Directional:
					directional_lights.push_back(light.buffer.get());
					cascade_shadowmaps.push_back(light.shadow_map->GetView(TextureViewDesc{
					    VK_IMAGE_VIEW_TYPE_2D_ARRAY,
					    VK_IMAGE_ASPECT_DEPTH_BIT,
					    0,
					    1,
					    0,
					    4}));
					break;
				case cmpt::LightType::Spot:
					spot_lights.push_back(light.buffer.get());
					shadowmaps.push_back(light.shadow_map->GetView(TextureViewDesc{
					    VK_IMAGE_VIEW_TYPE_2D,
					    VK_IMAGE_ASPECT_DEPTH_BIT,
					    0,
					    1,
					    0,
					    1}));
					break;
				case cmpt::LightType::Area:
					area_lights.push_back(light.buffer.get());
					break;
				default:
					break;
			}
		});*/

		struct
		{
			uint32_t point_light_count = 0;
		} push_constants;

		push_constants.point_light_count = static_cast<uint32_t>(point_lights.size());

		cmd_buffer.Bind(pso);
		cmd_buffer.Bind(
		    cmd_buffer.GetDescriptorState()
		        .Bind(0, 0, resource.GetTexture(visibility_buffer)->GetView(view_desc))
		        .Bind(0, 1, resource.GetTexture(shading)->GetView(view_desc))
		        .Bind(0, 2, resource.GetTexture(normal)->GetView(view_desc))
		        .Bind(0, 3, camera_buffer)
		        .Bind(0, 4, renderer.GetScene()->GetInstanceBuffer())
		        .Bind(0, 5, renderer.GetScene()->GetAssetManager().GetMeshletBuffer())
		        .Bind(0, 6, renderer.GetScene()->GetAssetManager().GetVertexBuffer())
		        .Bind(0, 7, renderer.GetScene()->GetAssetManager().GetMeshletVertexBuffer())
		        .Bind(0, 8, renderer.GetScene()->GetAssetManager().GetMeshletTriangleBuffer())
		        .Bind(0, 9, renderer.GetScene()->GetAssetManager().GetMaterialBuffer())
		        .Bind(0, 10, point_lights)

		        //.Bind(0, 5, directional_lights)
		        //.Bind(0, 6, spot_lights)
		        //.Bind(0, 8, area_lights)
		        //.Bind(0, 9, shadowmaps)
		        //.Bind(0, 10, cascade_shadowmaps)
		        //.Bind(0, 11, onmishadowmaps)
		        .Bind(0, 12, renderer.GetScene()->GetAssetManager().GetTextureViews())
		        .Bind(0, 13, renderer.GetSampler(SamplerType::TrilinearWarp)));
		cmd_buffer.PushConstants(VK_SHADER_STAGE_COMPUTE_BIT, &push_constants, sizeof(push_constants), 0);
		cmd_buffer.Dispatch((renderer.GetExtent().width + 32 - 1) / 32, (renderer.GetExtent().height + 32 - 1) / 32);
	});

	pass->BindImGui([=](ImGuiContext &, const RGResources &) {

	});

	builder.AddPass(std::move(pass));
}

}        // namespace Ilum