#include "ImGuiContext.hpp"

#include "Impl/imgui_impl_sdl.h"

#include "Impl/imgui_impl_vulkan.h"

#include "Core/Device/Instance.hpp"
#include "Core/Device/LogicalDevice.hpp"
#include "Core/Device/PhysicalDevice.hpp"
#include "Core/Device/Surface.hpp"
#include "Core/Device/Window.hpp"
#include "Core/Graphics/Command/CommandBuffer.hpp"
#include "Core/Graphics/Command/CommandPool.hpp"
#include "Core/Graphics/GraphicsContext.hpp"
#include "Core/Graphics/Image/Image2D.hpp"
#include "Core/Graphics/Image/ImageDepth.hpp"
#include "Core/Graphics/RenderPass/Framebuffer.hpp"
#include "Core/Graphics/RenderPass/RenderPass.hpp"
#include "Core/Graphics/RenderPass/RenderTarget.hpp"
#include "Core/Graphics/RenderPass/Swapchain.hpp"

namespace Ilum
{
inline VkDescriptorPool createDescriptorPool()
{
	VkDescriptorPoolSize pool_sizes[] =
	    {
	        {VK_DESCRIPTOR_TYPE_SAMPLER, 1000},
	        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000},
	        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000},
	        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000},
	        {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000},
	        {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000},
	        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000},
	        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000},
	        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000},
	        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000},
	        {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000}};
	VkDescriptorPoolCreateInfo pool_info = {};
	pool_info.sType                      = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	pool_info.flags                      = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
	pool_info.maxSets                    = 1000 * IM_ARRAYSIZE(pool_sizes);
	pool_info.poolSizeCount              = (uint32_t) IM_ARRAYSIZE(pool_sizes);
	pool_info.pPoolSizes                 = pool_sizes;
	VkDescriptorPool handle;
	vkCreateDescriptorPool(GraphicsContext::instance()->getLogicalDevice(), &pool_info, nullptr, &handle);

	return handle;
}

inline VkRenderPass createRenderPass(VkFormat format, VkSampleCountFlagBits samples)
{
	VkAttachmentDescription attachment = {};
	attachment.format                  = format;
	attachment.samples                 = samples;
	attachment.loadOp                  = VK_ATTACHMENT_LOAD_OP_CLEAR;
	attachment.storeOp                 = VK_ATTACHMENT_STORE_OP_STORE;
	attachment.stencilLoadOp           = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	attachment.stencilStoreOp          = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	attachment.initialLayout           = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	attachment.finalLayout             = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkAttachmentReference attachment_reference = {};
	attachment_reference.attachment            = 0;
	attachment_reference.layout                = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkSubpassDescription subpass = {};
	subpass.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments    = &attachment_reference;

	VkRenderPassCreateInfo create_info = {};
	create_info.sType                  = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	create_info.attachmentCount        = 1;
	create_info.pAttachments           = &attachment;
	create_info.subpassCount           = 1;
	create_info.pSubpasses             = &subpass;

	VkRenderPass handle = VK_NULL_HANDLE;
	vkCreateRenderPass(GraphicsContext::instance()->getLogicalDevice(), &create_info, nullptr, &handle);

	return handle;
}

ImGuiContext::ImGuiContext()
{
	std::vector<Attachment> attachments = {{0, "back_buffer", Attachment::Type::Swapchain, GraphicsContext::instance()->getSurface().getFormat().format}};
	std::vector<Subpass>    subpasses   = {{0, {0}}};
	VkRect2D                render_area = {};
	render_area.extent                  = GraphicsContext::instance()->getSwapchain().getExtent();

	m_render_target = createScope<RenderTarget>(std::move(attachments), std::move(subpasses), render_area);

	m_descriptor_pool = createDescriptorPool();

	ImGui_ImplVulkan_InitInfo init_info = {};
	init_info.Instance                  = GraphicsContext::instance()->getInstance();
	init_info.PhysicalDevice            = GraphicsContext::instance()->getPhysicalDevice();
	init_info.Device                    = GraphicsContext::instance()->getLogicalDevice();
	init_info.QueueFamily               = GraphicsContext::instance()->getLogicalDevice().getGraphicsFamily();
	init_info.Queue                     = GraphicsContext::instance()->getLogicalDevice().getGraphicsQueues()[0];
	init_info.PipelineCache             = GraphicsContext::instance()->getPipelineCache();
	init_info.DescriptorPool            = m_descriptor_pool;
	init_info.MinImageCount             = GraphicsContext::instance()->getSwapchain().getImageCount() - 1;
	init_info.ImageCount                = GraphicsContext::instance()->getSwapchain().getImageCount();

	ImGui_ImplSDL2_InitForVulkan(Window::instance()->getSDLHandle());
	ImGui_ImplVulkan_Init(&init_info, m_render_target->getRenderPass());

	// Upload fonts
	CommandBuffer command_buffer;
	command_buffer.begin();
	ImGui_ImplVulkan_CreateFontsTexture(command_buffer);
	command_buffer.end();
	command_buffer.submitIdle();
	ImGui_ImplVulkan_DestroyFontUploadObjects();

	// Event poll
	Window::instance()->Event_SDL += [](const SDL_Event &e) { ImGui_ImplSDL2_ProcessEvent(&e); };
}

ImGuiContext::~ImGuiContext()
{
	//vkDestroyRenderPass(GraphicsContext::instance()->getLogicalDevice(), m_render_pass, nullptr);
	vkDestroyDescriptorPool(GraphicsContext::instance()->getLogicalDevice(), m_descriptor_pool, nullptr);
}

void ImGuiContext::render(const CommandBuffer &command_buffer)
{
	command_buffer.beginRenderPass(*m_render_target);

	ImGui::Render();
	ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), command_buffer);

	command_buffer.endRenderPass();
}
}        // namespace Ilum