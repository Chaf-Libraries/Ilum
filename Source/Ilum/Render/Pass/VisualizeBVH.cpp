#include "VisualizeBVH.hpp"

#include <RHI/DescriptorState.hpp>

#include <Render/RGBuilder.hpp>
#include <Render/Renderer.hpp>

#include <Scene/Component/Camera.hpp>
#include <Scene/Component/MeshRenderer.hpp>
#include <Scene/Component/Transform.hpp>
#include <Scene/Scene.hpp>

#include <imgui.h>

#include <glm/gtc/type_ptr.hpp>

namespace Ilum
{
VisualizeBVH::VisualizeBVH() :
    RenderPass("VisualizeBVH")
{
}

void VisualizeBVH::Create(RGBuilder &builder)
{
	std::unique_ptr<RenderPass> pass = std::make_unique<VisualizeBVH>();

	// Render Target
	auto result = builder.CreateTexture(
	    "Result",
	    TextureDesc{
	        builder.GetRenderer().GetExtent().width,  /*width*/
	        builder.GetRenderer().GetExtent().height, /*height*/
	        1,                                        /*depth*/
	        1,                                        /*mips*/
	        1,                                        /*layers*/
	        VK_SAMPLE_COUNT_1_BIT,
	        VK_FORMAT_R8G8B8A8_UNORM,
	        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT},
	    TextureState{VK_IMAGE_USAGE_STORAGE_BIT});

	pass->AddResource(result);

	TextureViewDesc view_desc  = {};
	view_desc.aspect           = VK_IMAGE_ASPECT_COLOR_BIT;
	view_desc.view_type        = VK_IMAGE_VIEW_TYPE_2D;
	view_desc.base_array_layer = 0;
	view_desc.base_mip_level   = 0;
	view_desc.layer_count      = 1;
	view_desc.level_count      = 1;

	ShaderDesc shader  = {};
	shader.filename    = "./Source/Shaders/BVH/VisualizeBVH.hlsl";
	shader.entry_point = "main";
	shader.stage       = VK_SHADER_STAGE_COMPUTE_BIT;
	shader.type        = ShaderType::HLSL;

	PipelineState pso;
	pso
	    .SetName("VisualizeBVH")
	    .LoadShader(shader);

	struct PushConstants
	{
		glm::vec3 color = glm::vec3(0.27f, 0.65f, 0.73f);
	};

	std::shared_ptr<PushConstants> push_constants = std::make_shared<PushConstants>();

	pass->BindCallback([=](CommandBuffer &cmd_buffer, const RGResources &resource, Renderer &renderer) {
		if (!renderer.GetScene()->GetRegistry().valid(renderer.GetScene()->GetMainCamera()))
		{
			return;
		}

		cmpt::Camera &main_camera = renderer.GetScene()->GetRegistry().get<cmpt::Camera>(renderer.GetScene()->GetMainCamera());

		auto group = renderer.GetScene()->GetRegistry().group<cmpt::MeshRenderer>(entt::get<cmpt::Transform>);

		if (group.empty())
		{
			return;
		}

		std::vector<Buffer *> blas_buffer;
		std::vector<Buffer *> instance_buffer;
		blas_buffer.reserve(group.size());

		group.each([&](cmpt::MeshRenderer &mesh_renderer, cmpt::Transform &transform) {
			auto *mesh = mesh_renderer.GetMesh();
			if (mesh)
			{
				blas_buffer.push_back(&mesh->GetBLAS().GetBVHBuffer());
				instance_buffer.push_back(mesh_renderer.GetBuffer());
			}
		});

		cmd_buffer.Bind(pso);
		cmd_buffer.Bind(
		    cmd_buffer.GetDescriptorState()
		        .Bind(0, 0, main_camera.GetBuffer())
		        .Bind(0, 1, &renderer.GetScene()->GetTLAS().GetBVHBuffer())
		        .Bind(0, 2, blas_buffer)
		        .Bind(0, 3, instance_buffer)
		        .Bind(0, 4, resource.GetTexture(result)->GetView(view_desc)));
		cmd_buffer.PushConstants(VK_SHADER_STAGE_COMPUTE_BIT, push_constants.get(), sizeof(PushConstants), 0);
		cmd_buffer.Dispatch((renderer.GetExtent().width + 32 - 1) / 32, (renderer.GetExtent().height + 32 - 1) / 32);
	});

	pass->BindImGui([=](ImGuiContext &, const RGResources &) {
		ImGui::ColorEdit3("Background", glm::value_ptr(push_constants->color));
	});

	builder.AddPass(std::move(pass));
}
}        // namespace Ilum