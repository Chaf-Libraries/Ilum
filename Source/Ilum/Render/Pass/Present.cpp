#include "Present.hpp"
#include "../RGBuilder.hpp"

#include <RHI/ImGuiContext.hpp>

#include <imgui.h>

namespace Ilum
{
Present::Present() :
    RenderPass("Present")
{
}

void Present::Prepare(PipelineState &pso)
{
}

void Present::Create(RGBuilder &builder)
{
	std::unique_ptr<RenderPass> pass = std::make_unique<Present>();

	auto render_target = builder.CreateTexture(
	    "RenderTarget",
	    TextureDesc{0, 0, 0, 0, 0, VK_SAMPLE_COUNT_1_BIT, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT},
	    TextureState{VK_IMAGE_USAGE_SAMPLED_BIT});

	pass->AddResource(render_target);

	pass->BindCallback([=](CommandBuffer &cmd_buffer, PipelineState &pso, const RGResources &resource) {

	});

	pass->BindImGui([=](ImGuiContext &context, const RGResources &resources) {
		ImGui::Begin("Present");
		TextureViewDesc desc  = {};
		desc.view_type        = VK_IMAGE_VIEW_TYPE_2D;
		desc.aspect           = VK_IMAGE_ASPECT_COLOR_BIT;
		desc.base_mip_level   = 0;
		desc.base_array_layer = 0;
		desc.level_count      = resources.GetTexture(render_target)->GetMipLevels();
		desc.layer_count      = resources.GetTexture(render_target)->GetLayerCount();
		ImGui::Image(context.TextureID(resources.GetTexture(render_target)->GetView(desc)), ImGui::GetContentRegionAvail());
		ImGui::End();
	});

	builder.AddPass(std::move(pass));
}

}        // namespace Ilum