#include "ImGuiPass.hpp"

#include "Device/Instance.hpp"
#include "Device/LogicalDevice.hpp"
#include "Device/PhysicalDevice.hpp"
#include "Device/Window.hpp"
#include "Device/Surface.hpp"

#include "Graphics/Descriptor/DescriptorCache.hpp"
#include "Graphics/Descriptor/DescriptorPool.hpp"
#include "Graphics/GraphicsContext.hpp"
#include "Graphics/ImGui/Impl/imgui_impl_sdl.h"
#include "Graphics/ImGui/Impl/imgui_impl_vulkan.h"
#include "Graphics/Pipeline/PipelineState.hpp"
#include "Graphics/RenderPass/Swapchain.hpp"

namespace Ilum
{
ImGuiPass::ImGuiPass(const std::string &output_name, AttachmentState state):
    m_output(output_name), m_attachment_state(state)
{
}

void ImGuiPass::setupPipeline(PipelineState &state)
{
	state.addOutputAttachment(m_output, m_attachment_state);
	state.declareAttachment("output", GraphicsContext::instance()->getSurface().getFormat().format);
}

void ImGuiPass::resolveResources(ResolveState &resolve)
{
	
}

void ImGuiPass::render(RenderPassState &state)
{
	ImGui::Render();
	ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), state.command_buffer);
}
}        // namespace Ilum