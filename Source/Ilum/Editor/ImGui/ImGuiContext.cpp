#include "ImGuiContext.hpp"

#include <Core/Path.hpp>
#include <RenderCore/ShaderCompiler/ShaderCompiler.hpp>

#include <imgui.h>
#include <imgui_impl_glfw.h>

#include <IconsFontAwesome4.h>

namespace Ilum
{

	/*	platform_io.Renderer_CreateWindow  = RHI_Window_Create;
	platform_io.Renderer_DestroyWindow = RHI_Window_Destroy;
	platform_io.Renderer_SetWindowSize = RHI_Window_SetSize;
	platform_io.Renderer_RenderWindow  = RHI_Window_Render;
	platform_io.Renderer_SwapBuffers   = RHI_Window_Present;*/


ImGuiContext::ImGuiContext(RHIContext *context, Window *window) :
    p_context(context), p_window(window)
{
	ImGui::CreateContext();

	ImGui_ImplGlfw_InitForOther(window->GetHandle(), true);

	SetStyle();

	m_pipeline_state = context->CreatePipelineState();
	m_render_target  = context->CreateRenderTarget();
	m_sampler        = context->CreateSampler(SamplerDesc{
        RHIFilter::Linear,
        RHIFilter::Linear,
        RHIAddressMode::Clamp_To_Edge,
        RHIAddressMode::Clamp_To_Edge,
        RHIAddressMode::Clamp_To_Edge,
        RHIMipmapMode::Linear, 
		RHISamplerBorderColor::Float_Opaque_White});

	// Setup pipeline state
	DepthStencilState depth_stencil_state  = {};
	depth_stencil_state.depth_test_enable  = false;
	depth_stencil_state.depth_write_enable = false;
	depth_stencil_state.compare            = RHICompareOp::Always;

	RasterizationState rasterization_state = {};
	rasterization_state.cull_mode          = RHICullMode::None;
	rasterization_state.polygon_mode       = RHIPolygonMode::Solid;
	rasterization_state.front_face         = RHIFrontFace::Clockwise;

	InputAssemblyState input_assembly_state = {};
	input_assembly_state.topology           = RHIPrimitiveTopology::Triangle;

	BlendState blend_state = {};
	blend_state.enable     = false;
	blend_state.attachment_states.push_back(BlendState::AttachmentState{
	    true,
	    RHIBlendFactor::Src_Alpha,
	    RHIBlendFactor::One_Minus_Src_Alpha,
	    RHIBlendOp::Add,
	    RHIBlendFactor::One,
	    RHIBlendFactor::One_Minus_Src_Alpha,
	    RHIBlendOp::Add,
	    1 | 2 | 4 | 8});

	VertexInputState vertex_input_state = {};
	vertex_input_state.input_attributes = {
	    VertexInputState::InputAttribute{RHIVertexSemantics::Position, 0, 0, RHIFormat::R32G32_FLOAT, offsetof(ImDrawVert, pos)},
	    VertexInputState::InputAttribute{RHIVertexSemantics::Texcoord, 1, 0, RHIFormat::R32G32_FLOAT, offsetof(ImDrawVert, uv)},
	    VertexInputState::InputAttribute{RHIVertexSemantics::Texcoord, 2, 0, RHIFormat::R8G8B8A8_UNORM, offsetof(ImDrawVert, col)},
	};

	vertex_input_state.input_bindings = {
	    VertexInputState::InputBinding{0, sizeof(ImDrawVert), RHIVertexInputRate::Vertex}};

	std::vector<uint8_t> raw_shader;
	Path::GetInstance().Read("E:/Workspace/Ilum/Source/Shaders/ImGui.hlsl", raw_shader);

	std::string shader_source;
	shader_source.resize(raw_shader.size());
	std::memcpy(shader_source.data(), raw_shader.data(), raw_shader.size());

	ShaderDesc vertex_shader_desc  = {};
	vertex_shader_desc.entry_point = "VSmain";
	vertex_shader_desc.stage       = Ilum::RHIShaderStage::Vertex;
	vertex_shader_desc.source      = Ilum::ShaderSource::HLSL;
	vertex_shader_desc.target      = Ilum::ShaderTarget::SPIRV;
	vertex_shader_desc.code        = shader_source;

	ShaderDesc fragment_shader_desc  = {};
	fragment_shader_desc.entry_point = "PSmain";
	fragment_shader_desc.stage       = Ilum::RHIShaderStage::Fragment;
	fragment_shader_desc.source      = Ilum::ShaderSource::HLSL;
	fragment_shader_desc.target      = Ilum::ShaderTarget::SPIRV;
	fragment_shader_desc.code        = shader_source;

	Ilum::ShaderMeta vertex_meta   = {};
	Ilum::ShaderMeta fragment_meta = {};

	auto vertex_shader_spirv   = Ilum::ShaderCompiler::GetInstance().Compile(vertex_shader_desc, vertex_meta);
	auto fragment_shader_spirv = Ilum::ShaderCompiler::GetInstance().Compile(fragment_shader_desc, fragment_meta);

	m_vertex_shader   = p_context->CreateShader("VSmain", vertex_shader_spirv);
	m_fragment_shader = p_context->CreateShader("PSmain", fragment_shader_spirv);

	Ilum::ShaderMeta shader_meta = vertex_meta;
	shader_meta += fragment_meta;

	m_descriptor = p_context->CreateDescriptor(shader_meta);

	 m_pipeline_state->SetDepthStencilState(depth_stencil_state);
	 m_pipeline_state->SetRasterizationState(rasterization_state);
	m_pipeline_state->SetVertexInputState(vertex_input_state);
	m_pipeline_state->SetBlendState(blend_state);
	 m_pipeline_state->SetInputAssemblyState(input_assembly_state);
	m_pipeline_state->SetShader(RHIShaderStage::Vertex, m_vertex_shader.get());
	m_pipeline_state->SetShader(RHIShaderStage::Fragment, m_fragment_shader.get());

	// Font atlas
	unsigned char *pixels;
	int32_t        atlas_width, atlas_height, bpp;
	auto          &io = ImGui::GetIO();
	io.Fonts->GetTexDataAsRGBA32(&pixels, &atlas_width, &atlas_height, &bpp);

	const uint32_t size = atlas_width * atlas_height * bpp;

	auto buffer = p_context->CreateBuffer(static_cast<size_t>(size), RHIBufferUsage::Transfer, RHIMemoryUsage::CPU_TO_GPU);
	std::memcpy(buffer->Map(), pixels, size);
	buffer->Unmap();

	m_font_atlas = p_context->CreateTexture2D(atlas_width, atlas_height, RHIFormat::R8G8B8A8_UNORM, RHITextureUsage::ShaderResource | RHITextureUsage::Transfer, false);

	auto *cmd_buffer = p_context->CreateCommand(RHIQueueFamily::Graphics);
	cmd_buffer->Begin();
	cmd_buffer->ResourceStateTransition({TextureStateTransition{
	                                        m_font_atlas.get(),
	                                        RHIResourceState::Undefined,
	                                        RHIResourceState::TransferDest,
	                                        TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}}},
	                                    {});
	cmd_buffer->CopyBufferToTexture(buffer.get(), m_font_atlas.get(), 0, 0, 1);
	cmd_buffer->ResourceStateTransition({TextureStateTransition{
	                                        m_font_atlas.get(),
	                                        RHIResourceState::TransferDest,
	                                        RHIResourceState::ShaderResource,
	                                        TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}}},
	                                    {});
	cmd_buffer->End();
	auto *queue = p_context->GetQueue(RHIQueueFamily::Graphics);
	queue->Submit({cmd_buffer});
	auto fence = p_context->CreateFence();
	queue->Execute(fence.get());
	fence->Wait();

	io.Fonts->TexID = static_cast<ImTextureID>(m_font_atlas.get());

	// Create uniform buffer
	m_uniform_buffer = p_context->CreateBuffer(sizeof(m_constant_block), RHIBufferUsage::ConstantBuffer, RHIMemoryUsage::CPU_TO_GPU);
}

ImGuiContext::~ImGuiContext()
{
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
}

void ImGuiContext::NewFrame()
{
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();
}

void ImGuiContext::UpdateBuffers()
{
	ImDrawData *draw_data = ImGui::GetDrawData();

	size_t vertex_buffer_size = draw_data->TotalVtxCount * sizeof(ImDrawVert);
	size_t index_buffer_size  = draw_data->TotalIdxCount * sizeof(ImDrawIdx);

	if (vertex_buffer_size == 0 || index_buffer_size == 0)
	{
		return;
	}

	if (m_vertex_buffer == nullptr || m_vertex_count != draw_data->TotalVtxCount)
	{
		m_vertex_buffer.reset();
		m_vertex_buffer = p_context->CreateBuffer(vertex_buffer_size, RHIBufferUsage::Vertex, RHIMemoryUsage::CPU_TO_GPU);
		m_vertex_count  = draw_data->TotalVtxCount;
		m_vertex_buffer->Map();
	}

	if (m_index_buffer == nullptr || m_index_count < draw_data->TotalIdxCount)
	{
		m_index_buffer.reset();
		m_index_buffer = p_context->CreateBuffer(index_buffer_size, RHIBufferUsage::Index, RHIMemoryUsage::CPU_TO_GPU);
		m_vertex_count = draw_data->TotalIdxCount;
		m_index_buffer->Map();
	}

	ImDrawVert *vtx_dst = (ImDrawVert *) m_vertex_buffer->Map();
	ImDrawIdx  *idx_dst = (ImDrawIdx *) m_index_buffer->Map();

	for (int n = 0; n < draw_data->CmdListsCount; n++)
	{
		const ImDrawList *cmd_list = draw_data->CmdLists[n];
		memcpy(vtx_dst, cmd_list->VtxBuffer.Data, cmd_list->VtxBuffer.Size * sizeof(ImDrawVert));
		memcpy(idx_dst, cmd_list->IdxBuffer.Data, cmd_list->IdxBuffer.Size * sizeof(ImDrawIdx));
		vtx_dst += cmd_list->VtxBuffer.Size;
		idx_dst += cmd_list->IdxBuffer.Size;
	}

	m_vertex_buffer->Flush(0, m_vertex_buffer->GetDesc().size);
	m_index_buffer->Flush(0, m_index_buffer->GetDesc().size);
}

void ImGuiContext::Render()
{
	ImGui::Render();

	UpdateBuffers();

	ImDrawData *draw_data = ImGui::GetDrawData();

	int32_t fb_width  = (int32_t) (draw_data->DisplaySize.x * draw_data->FramebufferScale.x);
	int32_t fb_height = (int32_t) (draw_data->DisplaySize.y * draw_data->FramebufferScale.y);
	if (fb_width <= 0 || fb_height <= 0)
	{
		return;
	}

	m_constant_block.scale     = glm::vec2(2.0f / draw_data->DisplaySize.x, 2.0f / draw_data->DisplaySize.y);
	m_constant_block.translate = glm::vec2(-1.0f - draw_data->DisplayPos.x * m_constant_block.scale[0], -1.0f - draw_data->DisplayPos.y * m_constant_block.scale[1]);

	std::memcpy(m_uniform_buffer->Map(), &m_constant_block, sizeof(m_constant_block));

	m_descriptor->BindSampler("fontSampler", m_sampler.get());
	m_descriptor->BindBuffer("constant", m_uniform_buffer.get());

	m_render_target->Clear();
	ColorAttachment attachment = {};
	attachment.clear_value[3]  = 1.f;
	m_render_target->Add(p_context->GetBackBuffer(), TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}, ColorAttachment{});

	auto *cmd_buffer = p_context->CreateCommand(RHIQueueFamily::Graphics);

	cmd_buffer->Begin();

	cmd_buffer->ResourceStateTransition({Ilum::TextureStateTransition{
	                                        p_context->GetBackBuffer(),
	                                        Ilum::RHIResourceState::Undefined,
	                                        Ilum::RHIResourceState::RenderTarget,
	                                        Ilum::TextureRange{
	                                            Ilum::RHITextureDimension::Texture2D,
	                                            0, 1, 0, 1}}},
	                                    {});

	cmd_buffer->SetViewport(draw_data->DisplaySize.x, draw_data->DisplaySize.y);

	cmd_buffer->BeginRenderPass(m_render_target.get());

	int32_t global_vtx_offset = 0;
	int32_t global_idx_offset = 0;

	const ImVec2 &clip_off   = draw_data->DisplayPos;
	ImVec2        clip_scale = draw_data->FramebufferScale;

	void *current_texture = nullptr;

	for (int32_t i = 0; i < draw_data->CmdListsCount; i++)
	{
		cmd_buffer->BindVertexBuffer(m_vertex_buffer.get());
		cmd_buffer->BindIndexBuffer(m_index_buffer.get(), true);

		ImDrawList *cmd_list_imgui = draw_data->CmdLists[i];
		for (int32_t cmd_i = 0; cmd_i < cmd_list_imgui->CmdBuffer.Size; cmd_i++)
		{
			const ImDrawCmd *pcmd = &cmd_list_imgui->CmdBuffer[cmd_i];
			if (pcmd->UserCallback != nullptr)
			{
				pcmd->UserCallback(cmd_list_imgui, pcmd);
			}
			else
			{
				ImVec2 clip_min((pcmd->ClipRect.x - clip_off.x) * clip_scale.x, (pcmd->ClipRect.y - clip_off.y) * clip_scale.y);
				ImVec2 clip_max((pcmd->ClipRect.z - clip_off.x) * clip_scale.x, (pcmd->ClipRect.w - clip_off.y) * clip_scale.y);

				if (clip_min.x < 0.0f)
				{
					clip_min.x = 0.0f;
				}
				if (clip_min.y < 0.0f)
				{
					clip_min.y = 0.0f;
				}
				if (clip_max.x > fb_width)
				{
					clip_max.x = (float) fb_width;
				}
				if (clip_max.y > fb_height)
				{
					clip_max.y = (float) fb_height;
				}
				if (clip_max.x < clip_min.x || clip_max.y < clip_min.y)
				{
					continue;
				}

				cmd_buffer->SetScissor((uint32_t) (clip_max.x - clip_min.x), (uint32_t) (clip_max.y - clip_min.y), (int32_t) (clip_min.x), (int32_t) (clip_min.y));

				if (current_texture != pcmd->TextureId)
				{
					m_descriptor->BindTexture("fontTexture", static_cast<RHITexture *>(pcmd->TextureId), TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1});
					cmd_buffer->BindDescriptor(m_descriptor.get());
					cmd_buffer->BindPipelineState(m_pipeline_state.get());
					current_texture = pcmd->TextureId;
				}

				cmd_buffer->DrawIndexed(pcmd->ElemCount, 1, pcmd->IdxOffset + global_idx_offset, pcmd->VtxOffset + global_vtx_offset, 0);
			}
		}
		global_idx_offset += cmd_list_imgui->IdxBuffer.Size;
		global_vtx_offset += cmd_list_imgui->VtxBuffer.Size;
	}

	cmd_buffer->EndRenderPass();

	cmd_buffer->ResourceStateTransition({Ilum::TextureStateTransition{
	                                        p_context->GetBackBuffer(),
	                                        Ilum::RHIResourceState::RenderTarget,
	                                        Ilum::RHIResourceState::Present,
	                                        Ilum::TextureRange{
	                                            Ilum::RHITextureDimension::Texture2D,
	                                            0, 1, 0, 1}}},
	                                    {});

	cmd_buffer->End();

	p_context->GetQueue(RHIQueueFamily::Graphics)->Submit({cmd_buffer});
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
	config.MergeMode        = true;
	config.GlyphMinAdvanceX = 13.0f;
	io.Fonts->AddFontFromFileTTF("E:/Workspace/Ilum/Asset/Font/ArialUnicodeMS.ttf", 20.0f, NULL, io.Fonts->GetGlyphRangesChineseFull());
	io.Fonts->AddFontFromFileTTF("E:/Workspace/Ilum/Asset/Font/fontawesome-webfont.ttf", 15.0f, &config, icon_ranges);

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

void ImGuiContext::InitializePlatformInterface()
{
	ImGuiPlatformIO &platform_io       = ImGui::GetPlatformIO();
	platform_io.Renderer_CreateWindow  = RHI_Window_Create;
	platform_io.Renderer_DestroyWindow = RHI_Window_Destroy;
	platform_io.Renderer_SetWindowSize = RHI_Window_SetSize;
	platform_io.Renderer_RenderWindow  = RHI_Window_Render;
	platform_io.Renderer_SwapBuffers   = RHI_Window_Present;

}
}        // namespace Ilum