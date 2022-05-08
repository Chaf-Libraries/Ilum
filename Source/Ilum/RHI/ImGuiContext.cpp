#include "ImGuiContext.hpp"
#include "Command.hpp"
#include "Device.hpp"
#include "Texture.hpp"

#include <Core/Window.hpp>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <imnodes.h>

#include <IconsFontAwesome4.h>

#include <ImFileDialog/ImFileDialog.h>

namespace Ilum
{
ImGuiContext::ImGuiContext(Window *window, RHIDevice *device) :
    p_window(window),
    p_device(device)
{
	CreateDescriptorPool();
	CreateRenderPass();
	CreateFramebuffer();
	CreateSampler();

	ImGui::CreateContext();
	ImNodes::CreateContext();

	SetStyle();

	p_window->OnWindowSizeFunc += [this](int32_t width, int32_t height) { 
		if (width > 0 && height > 0)
		{
			CreateFramebuffer(); 
		}
	};

	ImGui_ImplGlfw_InitForVulkan(p_window->m_handle, true);

	ImGui_ImplVulkan_InitInfo init_info = {};
	init_info.Instance                  = p_device->GetVulkanInstance();
	init_info.PhysicalDevice            = p_device->GetPhysicalDevice();
	init_info.Device                    = p_device->GetDevice();
	init_info.QueueFamily               = p_device->GetGraphicsFamily();
	init_info.Queue                     = p_device->GetQueue(VK_QUEUE_GRAPHICS_BIT);
	;
	init_info.PipelineCache  = p_device->GetPipelineCache();
	init_info.DescriptorPool = m_descriptor_pool;
	init_info.MinImageCount  = static_cast<uint32_t>(p_device->GetSwapchainImages().size());
	init_info.ImageCount     = static_cast<uint32_t>(p_device->GetSwapchainImages().size());

	ImGui_ImplVulkan_Init(&init_info, m_render_pass);

	// Upload fonts
	auto &cmd_buffer = p_device->RequestCommandBuffer();

	vkDeviceWaitIdle(p_device->GetDevice());
	cmd_buffer.Begin();
	ImGui_ImplVulkan_CreateFontsTexture(cmd_buffer);
	cmd_buffer.End();
	p_device->SubmitIdle(cmd_buffer);
	ImGui_ImplVulkan_DestroyFontUploadObjects();

	// Setup Texture Create & Delete
	ifd::FileDialog::Instance().CreateTexture = [this](uint8_t *data, int w, int h, char fmt) -> void * {
		TextureDesc tex_desc = {};
		tex_desc.width       = static_cast<uint32_t>(w);
		tex_desc.height      = static_cast<uint32_t>(h);
		tex_desc.format      = fmt == 1 ? VK_FORMAT_R8G8B8A8_UNORM : VK_FORMAT_B8G8R8A8_UNORM;
		tex_desc.usage       = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

		BufferDesc buf_desc   = {};
		buf_desc.size         = static_cast<size_t>(w) * static_cast<size_t>(h) * 4ull;
		buf_desc.buffer_usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
		buf_desc.memory_usage = VMA_MEMORY_USAGE_CPU_TO_GPU;

		std::unique_ptr<Texture> tex = std::make_unique<Texture>(p_device, tex_desc);

		Buffer buffer(p_device, buf_desc);
		std::memcpy(buffer.Map(), data, buffer.GetSize());
		buffer.Flush(buf_desc.size);
		buffer.Unmap();

		BufferCopyInfo  buf_info = {&buffer, 0};
		TextureCopyInfo tex_info = {tex.get(), VkImageSubresourceLayers{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}};

		auto &cmd_buffer = p_device->RequestCommandBuffer();
		cmd_buffer.Begin();
		cmd_buffer.Transition(tex.get(), TextureState(), TextureState(VK_IMAGE_USAGE_TRANSFER_DST_BIT), VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
		cmd_buffer.CopyBufferToImage(buf_info, tex_info);
		cmd_buffer.Transition(tex.get(), TextureState(VK_IMAGE_USAGE_TRANSFER_DST_BIT), TextureState(VK_IMAGE_USAGE_SAMPLED_BIT), VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
		cmd_buffer.End();
		p_device->SubmitIdle(cmd_buffer);

		void *tex_id = TextureID(tex->GetView(TextureViewDesc{VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT}));

		m_filedialog_textures[tex_id] = std::move(tex);
		return tex_id;
	};
	ifd::FileDialog::Instance().DeleteTexture = [this](void *tex) {
		if (!m_destroy)
		{
			m_deprecated_filedialog_id.push_back(tex);
		}
	};
}

ImGuiContext::~ImGuiContext()
{
	vkDeviceWaitIdle(p_device->GetDevice());

	ImGui_ImplVulkan_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImNodes::DestroyContext();
	ImGui::DestroyContext();

	if (m_descriptor_pool)
	{
		vkDestroyDescriptorPool(p_device->GetDevice(), m_descriptor_pool, nullptr);
	}

	if (!m_frame_buffers.empty())
	{
		for (auto &frame_buffer : m_frame_buffers)
		{
			vkDestroyFramebuffer(p_device->GetDevice(), frame_buffer, nullptr);
		}
		m_frame_buffers.clear();
	}

	if (m_render_pass)
	{
		vkDestroyRenderPass(p_device->GetDevice(), m_render_pass, nullptr);
	}

	if (m_sampler)
	{
		vkDestroySampler(p_device->GetDevice(), m_sampler, nullptr);
	}

	m_texture_id_mapping.clear();
	m_filedialog_textures.clear();
	m_destroy = true;
}

void ImGuiContext::BeginFrame()
{
	if (m_texture_id_mapping.size() > 4000)
	{
		Flush();
	}

	ImGui_ImplVulkan_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

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

void ImGuiContext::Render()
{
	VkRect2D area            = {};
	area.extent.width        = p_window->m_width;
	area.extent.height       = p_window->m_height;
	VkClearValue clear_value = {};

	VkRenderPassBeginInfo begin_info = {};
	begin_info.sType                 = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	begin_info.renderPass            = m_render_pass;
	begin_info.renderArea            = area;
	begin_info.framebuffer           = m_frame_buffers[p_device->GetCurrentFrame()];
	begin_info.clearValueCount       = 1;
	begin_info.pClearValues          = &clear_value;

	auto &cmd_buffer = p_device->RequestCommandBuffer();
	cmd_buffer.Begin();
	vkCmdBeginRenderPass(cmd_buffer, &begin_info, VK_SUBPASS_CONTENTS_INLINE);
	ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd_buffer);
	cmd_buffer.EndRenderPass();
	cmd_buffer.End();
	p_device->Submit(cmd_buffer);
}

void ImGuiContext::EndFrame()
{
	ImGui::End();
	ImGuiIO &io    = ImGui::GetIO();
	io.DisplaySize = ImVec2(static_cast<float>(p_window->m_width), static_cast<float>(p_window->m_height));

	ImGui::EndFrame();

	ImGui::Render();

	if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
	{
		ImGui::UpdatePlatformWindows();
		ImGui::RenderPlatformWindowsDefault();
	}
}

void ImGuiContext::OpenFileDialog(const std::string &key, const std::string &title, const std::string &filter, bool open)
{
	if (open)
	{
		ifd::FileDialog::Instance().Open(key, title, filter);
	}
	else
	{
		ifd::FileDialog::Instance().Save(key, title, filter);
	}
}

void ImGuiContext::GetFileDialogResult(const std::string &key, std::function<void(const std::string &)> &&callback)
{
	if (ifd::FileDialog::Instance().IsDone(key))
	{
		if (ifd::FileDialog::Instance().HasResult())
		{
			std::string res = ifd::FileDialog::Instance().GetResult().u8string();
			callback(res);
		}
		ifd::FileDialog::Instance().Close();
	}
}

void *ImGuiContext::TextureID(VkImageView view)
{
	size_t hash = 0;
	HashCombine(hash, (uint64_t) view);

	if (m_texture_id_mapping.find(hash) == m_texture_id_mapping.end())
	{
		m_texture_id_mapping.emplace(hash, (VkDescriptorSet) ImGui_ImplVulkan_AddTexture(m_sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
	}

	return (ImTextureID) m_texture_id_mapping.at(hash);
}

void ImGuiContext::Flush()
{
	if (!m_texture_id_mapping.empty())
	{
		std::vector<VkDescriptorSet> descriptor_sets;
		descriptor_sets.reserve(m_texture_id_mapping.size());
		for (auto &[idx, set] : m_texture_id_mapping)
		{
			descriptor_sets.push_back(set);
		}
		vkFreeDescriptorSets(p_device->GetDevice(), m_descriptor_pool, static_cast<uint32_t>(descriptor_sets.size()), descriptor_sets.data());
		m_texture_id_mapping.clear();
	}
	if (!m_deprecated_filedialog_id.empty())
	{
		for (auto &id : m_deprecated_filedialog_id)
		{
			m_filedialog_textures.erase(id);
		}
		m_deprecated_filedialog_id.clear();
	}
}

void ImGuiContext::CreateDescriptorPool()
{
	if (m_descriptor_pool)
	{
		vkDestroyDescriptorPool(p_device->GetDevice(), m_descriptor_pool, nullptr);
		m_descriptor_pool = VK_NULL_HANDLE;
	}

	VkDescriptorPoolSize       pool_sizes[] = {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1}};
	VkDescriptorPoolCreateInfo pool_info    = {};
	pool_info.sType                         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	pool_info.flags                         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
	pool_info.maxSets                       = 4096 * IM_ARRAYSIZE(pool_sizes);
	pool_info.poolSizeCount                 = (uint32_t) IM_ARRAYSIZE(pool_sizes);
	pool_info.pPoolSizes                    = pool_sizes;
	vkCreateDescriptorPool(p_device->GetDevice(), &pool_info, nullptr, &m_descriptor_pool);
}

void ImGuiContext::CreateRenderPass()
{
	if (m_render_pass)
	{
		vkDestroyRenderPass(p_device->GetDevice(), m_render_pass, nullptr);
		m_render_pass = VK_NULL_HANDLE;
	}

	std::array<VkAttachmentDescription, 1> attachments = {};
	// Color attachment
	attachments[0].format         = p_device->GetSwapchainFormat();
	attachments[0].samples        = VK_SAMPLE_COUNT_1_BIT;
	attachments[0].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
	attachments[0].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
	attachments[0].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	attachments[0].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
	attachments[0].finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

	VkAttachmentReference colorReference = {};
	colorReference.attachment            = 0;
	colorReference.layout                = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkSubpassDescription subpassDescription    = {};
	subpassDescription.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpassDescription.colorAttachmentCount    = 1;
	subpassDescription.pColorAttachments       = &colorReference;
	subpassDescription.pDepthStencilAttachment = nullptr;
	subpassDescription.inputAttachmentCount    = 0;
	subpassDescription.pInputAttachments       = nullptr;
	subpassDescription.preserveAttachmentCount = 0;
	subpassDescription.pPreserveAttachments    = nullptr;
	subpassDescription.pResolveAttachments     = nullptr;

	// Subpass dependencies for layout transitions
	std::array<VkSubpassDependency, 2> dependencies;

	dependencies[0].srcSubpass      = VK_SUBPASS_EXTERNAL;
	dependencies[0].dstSubpass      = 0;
	dependencies[0].srcStageMask    = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
	dependencies[0].dstStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependencies[0].srcAccessMask   = VK_ACCESS_MEMORY_READ_BIT;
	dependencies[0].dstAccessMask   = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

	dependencies[1].srcSubpass      = 0;
	dependencies[1].dstSubpass      = VK_SUBPASS_EXTERNAL;
	dependencies[1].srcStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependencies[1].dstStageMask    = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
	dependencies[1].srcAccessMask   = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	dependencies[1].dstAccessMask   = VK_ACCESS_MEMORY_READ_BIT;
	dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

	VkRenderPassCreateInfo renderPassInfo = {};
	renderPassInfo.sType                  = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	renderPassInfo.attachmentCount        = static_cast<uint32_t>(attachments.size());
	renderPassInfo.pAttachments           = attachments.data();
	renderPassInfo.subpassCount           = 1;
	renderPassInfo.pSubpasses             = &subpassDescription;
	renderPassInfo.dependencyCount        = static_cast<uint32_t>(dependencies.size());
	renderPassInfo.pDependencies          = dependencies.data();

	vkCreateRenderPass(p_device->GetDevice(), &renderPassInfo, nullptr, &m_render_pass);
}

void ImGuiContext::CreateFramebuffer()
{
	if (!m_frame_buffers.empty())
	{
		for (auto &frame_buffer : m_frame_buffers)
		{
			vkDestroyFramebuffer(p_device->GetDevice(), frame_buffer, nullptr);
			frame_buffer = VK_NULL_HANDLE;
		}
		m_frame_buffers.clear();
	}

	VkFramebufferCreateInfo create_info = {};
	create_info.sType                   = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
	create_info.pNext                   = NULL;
	create_info.renderPass              = m_render_pass;
	create_info.attachmentCount         = 1;

	TextureViewDesc view_desc  = {};
	view_desc.aspect           = VK_IMAGE_ASPECT_COLOR_BIT;
	view_desc.view_type        = VK_IMAGE_VIEW_TYPE_2D;
	view_desc.base_mip_level   = 0;
	view_desc.base_array_layer = 0;
	view_desc.layer_count      = 1;
	view_desc.level_count      = 1;

	// Create frame buffers for every swap chain image
	m_frame_buffers.resize(p_device->GetSwapchainImages().size());
	for (uint32_t i = 0; i < m_frame_buffers.size(); i++)
	{
		VkImageView view         = p_device->GetSwapchainImages()[i]->GetView(view_desc);
		create_info.pAttachments = &view;
		create_info.width        = p_device->GetSwapchainImages()[i]->GetWidth();
		create_info.height       = p_device->GetSwapchainImages()[i]->GetHeight();
		create_info.layers       = 1;
		vkCreateFramebuffer(p_device->GetDevice(), &create_info, nullptr, &m_frame_buffers[i]);
	}
}

void ImGuiContext::CreateSampler()
{
	VkSamplerCreateInfo create_info = {};
	create_info.sType               = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	create_info.minFilter           = VK_FILTER_LINEAR;
	create_info.magFilter           = VK_FILTER_LINEAR;
	create_info.mipmapMode          = VK_SAMPLER_MIPMAP_MODE_LINEAR;
	create_info.addressModeU        = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	create_info.addressModeV        = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	create_info.addressModeW        = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	create_info.mipLodBias          = 0.f;
	create_info.minLod              = 0.f;
	create_info.maxLod              = 10.f;

	vkCreateSampler(p_device->GetDevice(), &create_info, nullptr, &m_sampler);
}

void ImGuiContext::SetStyle()
{
	ImGuiIO &io = ImGui::GetIO();
	(void) io;
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;        // Enable Keyboard Controls
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;         // Enable Gamepad Controls
	io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;            // Enable Docking
	io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;          // Enable Multi-Viewport / Platform Windows

	// Set fonts
	static const ImWchar icon_ranges[] = {ICON_MIN_FA, ICON_MAX_FA, 0};
	ImFontConfig         config;
	config.MergeMode = true;
	config.GlyphMinAdvanceX = 13.0f;
	io.Fonts->AddFontFromFileTTF("Asset/Font/ArialUnicodeMS.ttf", 25.0f, NULL, io.Fonts->GetGlyphRangesChineseFull());
	io.Fonts->AddFontFromFileTTF("Asset/Font/fontawesome-webfont.ttf", 20.0f, &config, icon_ranges);

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

}        // namespace Ilum