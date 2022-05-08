#include "Present.hpp"
#include "../RGBuilder.hpp"
#include "../Renderer.hpp"

#include <RHI/ImGuiContext.hpp>

#include <imgui.h>

namespace Ilum
{
Present::Present() :
    RenderPass("Present")
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

	pass->BindCallback([=](CommandBuffer &cmd_buffer, const RGResources &resource, Renderer & renderer) {
		renderer.SetPresent(resource.GetTexture(render_target));
	});

	pass->BindImGui([=](ImGuiContext &context, const RGResources &resources) {
	});

	builder.AddPass(std::move(pass));
}

}        // namespace Ilum