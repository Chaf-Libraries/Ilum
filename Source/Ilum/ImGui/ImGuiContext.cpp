#include "ImGuiContext.hpp"

#include <imgui_impl_sdl.h>
#include <imgui_impl_vulkan.h>

#include "Device/Instance.hpp"
#include "Device/LogicalDevice.hpp"
#include "Device/PhysicalDevice.hpp"
#include "Device/Surface.hpp"
#include "Device/Swapchain.hpp"
#include "Device/Window.hpp"

#include "Graphics/Command/CommandBuffer.hpp"
#include "Graphics/Command/CommandPool.hpp"
#include "Graphics/GraphicsContext.hpp"
#include "Graphics/Synchronization/Queue.hpp"
#include "Graphics/Synchronization/QueueSystem.hpp"
#include "Graphics/Vulkan/VK_Debugger.h"

#include "Renderer/RenderGraph/RenderGraph.hpp"
#include "Renderer/RenderPass/ImGuiPass.hpp"
#include "Renderer/Renderer.hpp"

#include "Loader/ImageLoader/Bitmap.hpp"
#include "Loader/ImageLoader/ImageLoader.hpp"

#include <ImFileDialog.h>

namespace Ilum
{
scope<ImGuiContext> ImGuiContext::s_instance = nullptr;
bool                ImGuiContext::s_enable   = false;

inline VkDescriptorPool createDescriptorPool()
{
	VkDescriptorPoolSize pool_sizes[] =
	    {
	        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1}};
	VkDescriptorPoolCreateInfo pool_info = {};
	pool_info.sType                      = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	pool_info.flags                      = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
	pool_info.maxSets                    = 4096 * IM_ARRAYSIZE(pool_sizes);
	pool_info.poolSizeCount              = (uint32_t) IM_ARRAYSIZE(pool_sizes);
	pool_info.pPoolSizes                 = pool_sizes;
	VkDescriptorPool handle;
	vkCreateDescriptorPool(GraphicsContext::instance()->getLogicalDevice(), &pool_info, nullptr, &handle);

	return handle;
}

ImGuiContext::ImGuiContext()
{
	// Event poll
	Window::instance()->Event_SDL += [this](const SDL_Event &e) {
		if (Renderer::instance()->hasImGui() && s_enable)
		{
			ImGui_ImplSDL2_ProcessEvent(&e);
		}
	};

	// Recreate everything when rebuild swapchain
	GraphicsContext::instance()->Swapchain_Rebuild_Event += [this]() { releaseResource(); createResouce(); };
}

void ImGuiContext::createResouce()
{
	if (!Renderer::instance()->hasImGui())
	{
		return;
	}

	ImGui::CreateContext();

	s_instance->m_texture_id_mapping.clear();

	// Config style
	setDarkMode();

	s_instance->m_descriptor_pool = createDescriptorPool();

	// Setting file dialog
	ifd::FileDialog::Instance().CreateTexture = [](uint8_t *data, int w, int h, char fmt) -> void * {
		Bitmap bitmap;
		Image  image;

		bitmap.data.resize(static_cast<size_t>(w) * static_cast<size_t>(h) * 4ull);
		bitmap.format = fmt == 1 ? VK_FORMAT_R8G8B8A8_UNORM : VK_FORMAT_B8G8R8A8_UNORM;
		bitmap.width  = static_cast<uint32_t>(w);
		bitmap.height = static_cast<uint32_t>(h);

		std::memcpy(bitmap.data.data(), data, static_cast<size_t>(w) * static_cast<size_t>(h) * 4ull);
		ImageLoader::loadImage(image, bitmap, true);

		auto texure_id = (VkDescriptorSet) ImGui_ImplVulkan_AddTexture(Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp), image.getView(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		s_instance->m_filedialog_image_cache.emplace((VkDescriptorSet) texure_id, std::move(image));
		static int i = 0;
		VK_Debugger::setName(GraphicsContext::instance()->getLogicalDevice(), (VkDescriptorSet) texure_id, std::to_string(i++).c_str());

		return texure_id;
	};

	ifd::FileDialog::Instance().DeleteTexture = [](void *tex) {
		s_instance->m_deprecated_descriptor_sets.push_back((VkDescriptorSet) tex);
	};

	ImGui_ImplVulkan_InitInfo init_info = {};
	init_info.Instance                  = GraphicsContext::instance()->getInstance();
	init_info.PhysicalDevice            = GraphicsContext::instance()->getPhysicalDevice();
	init_info.Device                    = GraphicsContext::instance()->getLogicalDevice();
	init_info.QueueFamily               = GraphicsContext::instance()->getLogicalDevice().getGraphicsFamily();
	init_info.Queue                     = GraphicsContext::instance()->getLogicalDevice().getGraphicsQueues()[0];
	init_info.PipelineCache             = GraphicsContext::instance()->getPipelineCache();
	init_info.DescriptorPool            = s_instance->m_descriptor_pool;
	init_info.MinImageCount             = GraphicsContext::instance()->getSwapchain().getImageCount() - 1;
	init_info.ImageCount                = GraphicsContext::instance()->getSwapchain().getImageCount();

	ImGui_ImplSDL2_InitForVulkan(Window::instance()->getSDLHandle());
	ImGui_ImplVulkan_Init(&init_info, Renderer::instance()->getRenderGraph()->getNode<pass::ImGuiPass>().pass_native.render_pass);

	// Upload fonts
	CommandBuffer command_buffer;
	command_buffer.begin();
	ImGui_ImplVulkan_CreateFontsTexture(command_buffer);
	command_buffer.end();
	//GraphicsContext::instance()->getQueueSystem().acquire(QueueUsage::Transfer)->submitIdle(command_buffer);
	command_buffer.submitIdle();
	ImGui_ImplVulkan_DestroyFontUploadObjects();

	s_enable = true;
}

void ImGuiContext::releaseResource()
{
	if (s_enable)
	{
		vkQueueWaitIdle(GraphicsContext::instance()->getLogicalDevice().getGraphicsQueues()[GraphicsContext::instance()->getFrameIndex() % GraphicsContext::instance()->getLogicalDevice().getGraphicsQueues().size()]);
		ImGui_ImplVulkan_Shutdown();
		ImGui_ImplSDL2_Shutdown();
		ImGui::DestroyContext();

		// Release resource
		vkDestroyDescriptorPool(GraphicsContext::instance()->getLogicalDevice(), s_instance->m_descriptor_pool, nullptr);
		s_instance->m_texture_id_mapping.clear();
		s_instance->m_filedialog_image_cache.clear();
		s_instance->m_deprecated_descriptor_sets.clear();

		s_enable = false;
	}
}

void *ImGuiContext::textureID(const Image &image, const Sampler &sampler)
{
	return textureID(image.getView(), sampler);
}

void *ImGuiContext::textureID(const VkImageView &view, const Sampler &sampler)
{
	size_t hash = 0;
	hash_combine(hash, (uint64_t) view);
	hash_combine(hash, (uint64_t) sampler.getSampler());

	if (s_instance->m_texture_id_mapping.find(hash) == s_instance->m_texture_id_mapping.end())
	{
		s_instance->m_texture_id_mapping.emplace(hash, (VkDescriptorSet) ImGui_ImplVulkan_AddTexture(sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
	}

	return (ImTextureID) s_instance->m_texture_id_mapping.at(hash);
}

void ImGuiContext::flush()
{
	if (s_instance && !s_instance->m_texture_id_mapping.empty())
	{
		std::vector<VkDescriptorSet> descriptor_sets;
		descriptor_sets.reserve(s_instance->m_texture_id_mapping.size());
		for (auto &[idx, set] : s_instance->m_texture_id_mapping)
		{
			descriptor_sets.push_back(set);
		}
		vkFreeDescriptorSets(GraphicsContext::instance()->getLogicalDevice(), s_instance->m_descriptor_pool, static_cast<uint32_t>(descriptor_sets.size()), descriptor_sets.data());
		s_instance->m_texture_id_mapping.clear();
		s_instance->m_filedialog_image_cache.clear();
	}

	if (s_instance && !s_instance->m_deprecated_descriptor_sets.empty())
	{
		for (auto &set : s_instance->m_deprecated_descriptor_sets)
		{
			s_instance->m_filedialog_image_cache.erase(set);
		}
		vkFreeDescriptorSets(GraphicsContext::instance()->getLogicalDevice(), s_instance->m_descriptor_pool, static_cast<uint32_t>(s_instance->m_deprecated_descriptor_sets.size()), s_instance->m_deprecated_descriptor_sets.data());
		s_instance->m_deprecated_descriptor_sets.clear();
	}
}

bool ImGuiContext::enable() const
{
	return s_enable;
}

void ImGuiContext::setDarkMode()
{
	ImGuiIO &io = ImGui::GetIO();
	(void) io;
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;        // Enable Keyboard Controls
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;         // Enable Gamepad Controls
	io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;            // Enable Docking
	io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;          // Enable Multi-Viewport / Platform Windows

	// Set a better fonts
	io.Fonts->AddFontFromFileTTF((std::string(PROJECT_SOURCE_DIR) + "/Asset/Font/arialbd.ttf").c_str(), 15.0f);

	// When viewports are enabled we tweak WindowRounding/WindowBg so platform windows can look identical to regular ones.
	ImGuiStyle &style  = ImGui::GetStyle();
	auto        colors = style.Colors;

	if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
	{
		style.WindowRounding              = 0.0f;
		style.Colors[ImGuiCol_WindowBg].w = 1.0f;
	}

	colors[ImGuiCol_Text]                  = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
	colors[ImGuiCol_TextDisabled]          = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);
	colors[ImGuiCol_WindowBg]              = ImVec4(0.06f, 0.06f, 0.06f, 0.94f);
	colors[ImGuiCol_ChildBg]               = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
	colors[ImGuiCol_PopupBg]               = ImVec4(0.08f, 0.08f, 0.08f, 0.94f);
	colors[ImGuiCol_Border]                = ImVec4(0.43f, 0.43f, 0.50f, 0.50f);
	colors[ImGuiCol_BorderShadow]          = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
	colors[ImGuiCol_FrameBg]               = ImVec4(0.44f, 0.44f, 0.44f, 0.60f);
	colors[ImGuiCol_FrameBgHovered]        = ImVec4(0.57f, 0.57f, 0.57f, 0.70f);
	colors[ImGuiCol_FrameBgActive]         = ImVec4(0.76f, 0.76f, 0.76f, 0.80f);
	colors[ImGuiCol_TitleBg]               = ImVec4(0.04f, 0.04f, 0.04f, 1.00f);
	colors[ImGuiCol_TitleBgActive]         = ImVec4(0.16f, 0.16f, 0.16f, 1.00f);
	colors[ImGuiCol_TitleBgCollapsed]      = ImVec4(0.00f, 0.00f, 0.00f, 0.60f);
	colors[ImGuiCol_MenuBarBg]             = ImVec4(0.14f, 0.14f, 0.14f, 1.00f);
	colors[ImGuiCol_ScrollbarBg]           = ImVec4(0.02f, 0.02f, 0.02f, 0.53f);
	colors[ImGuiCol_ScrollbarGrab]         = ImVec4(0.31f, 0.31f, 0.31f, 1.00f);
	colors[ImGuiCol_ScrollbarGrabHovered]  = ImVec4(0.41f, 0.41f, 0.41f, 1.00f);
	colors[ImGuiCol_ScrollbarGrabActive]   = ImVec4(0.51f, 0.51f, 0.51f, 1.00f);
	colors[ImGuiCol_CheckMark]             = ImVec4(0.13f, 0.75f, 0.55f, 0.80f);
	colors[ImGuiCol_SliderGrab]            = ImVec4(0.13f, 0.75f, 0.75f, 0.80f);
	colors[ImGuiCol_SliderGrabActive]      = ImVec4(0.13f, 0.75f, 1.00f, 0.80f);
	colors[ImGuiCol_Button]                = ImVec4(0.13f, 0.75f, 0.55f, 0.40f);
	colors[ImGuiCol_ButtonHovered]         = ImVec4(0.13f, 0.75f, 0.75f, 0.60f);
	colors[ImGuiCol_ButtonActive]          = ImVec4(0.13f, 0.75f, 1.00f, 0.80f);
	colors[ImGuiCol_Header]                = ImVec4(0.13f, 0.75f, 0.55f, 0.40f);
	colors[ImGuiCol_HeaderHovered]         = ImVec4(0.13f, 0.75f, 0.75f, 0.60f);
	colors[ImGuiCol_HeaderActive]          = ImVec4(0.13f, 0.75f, 1.00f, 0.80f);
	colors[ImGuiCol_Separator]             = ImVec4(0.13f, 0.75f, 0.55f, 0.40f);
	colors[ImGuiCol_SeparatorHovered]      = ImVec4(0.13f, 0.75f, 0.75f, 0.60f);
	colors[ImGuiCol_SeparatorActive]       = ImVec4(0.13f, 0.75f, 1.00f, 0.80f);
	colors[ImGuiCol_ResizeGrip]            = ImVec4(0.13f, 0.75f, 0.55f, 0.40f);
	colors[ImGuiCol_ResizeGripHovered]     = ImVec4(0.13f, 0.75f, 0.75f, 0.60f);
	colors[ImGuiCol_ResizeGripActive]      = ImVec4(0.13f, 0.75f, 1.00f, 0.80f);
	colors[ImGuiCol_Tab]                   = ImVec4(0.13f, 0.75f, 0.55f, 0.80f);
	colors[ImGuiCol_TabHovered]            = ImVec4(0.13f, 0.75f, 0.75f, 0.80f);
	colors[ImGuiCol_TabActive]             = ImVec4(0.13f, 0.75f, 1.00f, 0.80f);
	colors[ImGuiCol_TabUnfocused]          = ImVec4(0.18f, 0.18f, 0.18f, 1.00f);
	colors[ImGuiCol_TabUnfocusedActive]    = ImVec4(0.36f, 0.36f, 0.36f, 0.54f);
	colors[ImGuiCol_DockingPreview]        = ImVec4(0.13f, 0.75f, 0.55f, 0.80f);
	colors[ImGuiCol_DockingEmptyBg]        = ImVec4(0.13f, 0.13f, 0.13f, 0.80f);
	colors[ImGuiCol_PlotLines]             = ImVec4(0.61f, 0.61f, 0.61f, 1.00f);
	colors[ImGuiCol_PlotLinesHovered]      = ImVec4(1.00f, 0.43f, 0.35f, 1.00f);
	colors[ImGuiCol_PlotHistogram]         = ImVec4(0.90f, 0.70f, 0.00f, 1.00f);
	colors[ImGuiCol_PlotHistogramHovered]  = ImVec4(1.00f, 0.60f, 0.00f, 1.00f);
	colors[ImGuiCol_TableHeaderBg]         = ImVec4(0.19f, 0.19f, 0.20f, 1.00f);
	colors[ImGuiCol_TableBorderStrong]     = ImVec4(0.31f, 0.31f, 0.35f, 1.00f);
	colors[ImGuiCol_TableBorderLight]      = ImVec4(0.23f, 0.23f, 0.25f, 1.00f);
	colors[ImGuiCol_TableRowBg]            = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
	colors[ImGuiCol_TableRowBgAlt]         = ImVec4(1.00f, 1.00f, 1.00f, 0.07f);
	colors[ImGuiCol_TextSelectedBg]        = ImVec4(0.26f, 0.59f, 0.98f, 0.35f);
	colors[ImGuiCol_DragDropTarget]        = ImVec4(1.00f, 1.00f, 0.00f, 0.90f);
	colors[ImGuiCol_NavHighlight]          = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
	colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
	colors[ImGuiCol_NavWindowingDimBg]     = ImVec4(0.80f, 0.80f, 0.80f, 0.20f);
	colors[ImGuiCol_ModalWindowDimBg]      = ImVec4(0.80f, 0.80f, 0.80f, 0.35f);

	style.WindowPadding     = ImVec2(8.00f, 8.00f);
	style.FramePadding      = ImVec2(5.00f, 2.00f);
	style.CellPadding       = ImVec2(6.00f, 6.00f);
	style.ItemSpacing       = ImVec2(6.00f, 6.00f);
	style.ItemInnerSpacing  = ImVec2(6.00f, 6.00f);
	style.TouchExtraPadding = ImVec2(0.00f, 0.00f);
	style.IndentSpacing     = 25;
	style.ScrollbarSize     = 15;
	style.GrabMinSize       = 10;
	style.WindowBorderSize  = 1;
	style.ChildBorderSize   = 1;
	style.PopupBorderSize   = 1;
	style.FrameBorderSize   = 1;
	style.TabBorderSize     = 1;
	style.WindowRounding    = 7;
	style.ChildRounding     = 4;
	style.FrameRounding     = 3;
	style.PopupRounding     = 4;
	style.ScrollbarRounding = 9;
	style.GrabRounding      = 3;
	style.LogSliderDeadzone = 4;
	style.TabRounding       = 4;
}

void ImGuiContext::initialize()
{
	s_instance = createScope<ImGuiContext>();

	createResouce();
}

void ImGuiContext::destroy()
{
	releaseResource();

	s_instance.reset();
}

void ImGuiContext::begin()
{
	if (!Renderer::instance()->hasImGui())
	{
		return;
	}

	if (s_instance->m_texture_id_mapping.size() + s_instance->m_filedialog_image_cache.size() > 4000)
	{
		flush();
	}

	ImGui_ImplVulkan_NewFrame();
	ImGui_ImplSDL2_NewFrame();
	ImGui::NewFrame();

	beginDockingSpace();
}

void ImGuiContext::render(const CommandBuffer &command_buffer)
{
	ImGui::Render();
	ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), command_buffer);

	if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
	{
		ImGui::UpdatePlatformWindows();
		ImGui::RenderPlatformWindowsDefault();
	}
}

void ImGuiContext::end()
{
	if (!Renderer::instance()->hasImGui())
	{
		return;
	}

	endDockingSpace();

	ImGui::EndFrame();
}

void ImGuiContext::beginDockingSpace()
{
	ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDocking;
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

void ImGuiContext::endDockingSpace()
{
	ImGui::End();
	ImGuiIO &io    = ImGui::GetIO();
	io.DisplaySize = ImVec2(static_cast<float>(Window::instance()->getWidth()), static_cast<float>(Window::instance()->getHeight()));
}
}        // namespace Ilum