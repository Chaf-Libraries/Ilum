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
	        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000}};
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

ImGuiContext::ImGuiContext(Context *context) :
    TSubsystem<ImGuiContext>(context)
{
	// Event poll
	Window::instance()->Event_SDL += [](const SDL_Event &e) { ImGui_ImplSDL2_ProcessEvent(&e); };

	// Recreate everything when rebuild swapchain
	GraphicsContext::instance()->Swapchain_Rebuild_Event += [this]() { onShutdown(); onInitialize(); };
}

ImGuiContext::~ImGuiContext()
{
}

bool ImGuiContext::onInitialize()
{
	ImGui::CreateContext();

	// Config style
	config();

	std::vector<Attachment> attachments = {{0, "back_buffer", Attachment::Type::Swapchain, GraphicsContext::instance()->getSurface().getFormat().format}};
	std::vector<Subpass>    subpasses   = {{0, {0}}};
	VkRect2D                render_area = {};
	render_area.extent                  = GraphicsContext::instance()->getSwapchain().getExtent();

	m_render_target = createScope<RenderTarget>(std::move(attachments), std::move(subpasses), render_area);

	m_descriptor_pool = createDescriptorPool();
	for (auto &command_buffer : m_command_buffers)
	{
		command_buffer = createScope<CommandBuffer>();
	}

	VkSemaphoreCreateInfo create_info = {};
	create_info.sType                 = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
	for (auto &semaphore : m_render_complete)
	{
		vkCreateSemaphore(GraphicsContext::instance()->getLogicalDevice(), &create_info, nullptr, &semaphore);
	}

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
	uploadFontsData();

	return true;
}

void ImGuiContext::onPreTick()
{
	ImGui_ImplVulkan_NewFrame();
	ImGui_ImplSDL2_NewFrame();
	ImGui::NewFrame();

	// Begin docking space
	ImGuiWindowFlags window_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking;
	window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
	window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
	ImGuiViewport *viewport = ImGui::GetMainViewport();
	ImGui::SetNextWindowPos(viewport->WorkPos);
	ImGui::SetNextWindowSize(viewport->WorkSize);
	ImGui::SetNextWindowViewport(viewport->ID);
	ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
	ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
	ImGui::Begin("DockSpace", (bool *) 1, window_flags);
	ImGui::PopStyleVar();
	ImGui::PopStyleVar(2);

	ImGuiIO &io = ImGui::GetIO();
	if (io.ConfigFlags & ImGuiConfigFlags_DockingEnable)
	{
		ImGuiID dockspace_id = ImGui::GetID("DockSpace");
		ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_None);
	}
}

void ImGuiContext::onPostTick()
{
	// End docking space
	ImGui::End();

	ImGuiIO &io    = ImGui::GetIO();
	io.DisplaySize = ImVec2(static_cast<float>(Window::instance()->getWidth()), static_cast<float>(Window::instance()->getHeight()));

	// Render UI
	auto &command_buffer = m_command_buffers[GraphicsContext::instance()->getSwapchain().getActiveImageIndex()];
	command_buffer->begin();
	command_buffer->beginRenderPass(*m_render_target);
	ImGui::Render();
	ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), *command_buffer);
	command_buffer->endRenderPass();
	command_buffer->end();

	// Submit command buffer
	command_buffer->submit(
	    GraphicsContext::instance()->getRenderCompleteSemaphore(),
	    m_render_complete[GraphicsContext::instance()->getFrameIndex()],
	    VK_NULL_HANDLE,
	    {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT},
	    0);

	ImGui::EndFrame();

	if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
	{
		ImGui::UpdatePlatformWindows();
		ImGui::RenderPlatformWindowsDefault();
	}
}

void ImGuiContext::onShutdown()
{
	vkQueueWaitIdle(GraphicsContext::instance()->getLogicalDevice().getGraphicsQueues()[GraphicsContext::instance()->getFrameIndex() % GraphicsContext::instance()->getLogicalDevice().getGraphicsQueues().size()]);
	ImGui_ImplVulkan_Shutdown();
	ImGui_ImplSDL2_Shutdown();
	ImGui::DestroyContext();

	// Release resource
	vkDestroyDescriptorPool(GraphicsContext::instance()->getLogicalDevice(), m_descriptor_pool, nullptr);
	for (auto &semaphore : m_render_complete)
	{
		vkDestroySemaphore(GraphicsContext::instance()->getLogicalDevice(), semaphore, nullptr);
	}
}

void ImGuiContext::render(const CommandBuffer &command_buffer)
{
	command_buffer.beginRenderPass(*m_render_target);

	ImGui::Render();
	ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), command_buffer);

	command_buffer.endRenderPass();
}

const VkSemaphore &ImGuiContext::getRenderCompleteSemaphore() const
{
	return m_render_complete[GraphicsContext::instance()->getFrameIndex()];
}

void *ImGuiContext::textureID(const Image2D *image)
{
	if (m_texture_id_mapping.find(&image->getDescriptor()) != m_texture_id_mapping.end())
	{
		return (ImTextureID) m_texture_id_mapping.at(&image->getDescriptor());
	}

	return (ImTextureID) ImGui_ImplVulkan_AddTexture(image->getSampler(), image->getView(), image->getImageLayout());
}

void ImGuiContext::config()
{
	ImGuiIO &io = ImGui::GetIO();
	(void) io;
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;        // Enable Keyboard Controls
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;         // Enable Gamepad Controls
	io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;            // Enable Docking
	io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;          // Enable Multi-Viewport / Platform Windows

	// Setup Dear ImGui style
	ImGui::StyleColorsDark();

	// Set a better fonts
	io.Fonts->AddFontFromFileTTF((std::string(PROJECT_SOURCE_DIR) + "/Asset/Font/arialbd.ttf").c_str(), 15.0f);

	// When viewports are enabled we tweak WindowRounding/WindowBg so platform windows can look identical to regular ones.
	ImGuiStyle &style = ImGui::GetStyle();
	if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
	{
		style.WindowRounding              = 0.0f;
		style.Colors[ImGuiCol_WindowBg].w = 1.0f;
	}
}

void ImGuiContext::uploadFontsData()
{
	CommandBuffer command_buffer;
	command_buffer.begin();
	ImGui_ImplVulkan_CreateFontsTexture(command_buffer);
	command_buffer.end();
	command_buffer.submitIdle();
	ImGui_ImplVulkan_DestroyFontUploadObjects();
}
}        // namespace Ilum