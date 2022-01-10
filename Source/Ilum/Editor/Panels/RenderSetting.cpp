#include "RenderSetting.hpp"

#include "Renderer/Renderer.hpp"

#include "ImGui/ImGuiContext.hpp"

#include "File/FileSystem.hpp"

#include <imgui.h>
#include <imgui_internal.h>

namespace Ilum::panel
{
template <typename Callback>
inline void draw_node(const std::string &name, Callback callback)
{
	const ImGuiTreeNodeFlags tree_node_flags          = ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_Framed | ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_AllowItemOverlap | ImGuiTreeNodeFlags_FramePadding;
	ImVec2                   content_region_available = ImGui::GetContentRegionAvail();

	ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2{4, 4});
	float line_height = GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f;
	bool  open        = ImGui::TreeNodeEx(name.c_str(), tree_node_flags, name.c_str());
	ImGui::PopStyleVar();

	if (open)
	{
		callback();
		ImGui::TreePop();
	}
}

RenderSetting::RenderSetting()
{
	m_name = "Render Setting";
}

void RenderSetting::draw(float delta_time)
{
	ImGui::Begin("Render Setting", &active);

	const char *const render_mode[]       = {"Polygon", "Wire Frame", "Point Cloud"};
	int               current_render_mode = static_cast<int>(Renderer::instance()->Render_Mode);

	if (ImGui::Combo("Render Mode", &current_render_mode, render_mode, 3))
	{
		Renderer::instance()->Render_Mode = static_cast<Renderer::RenderMode>(current_render_mode);
		Renderer::instance()->update();
	}

	draw_node("Culling", []() {
		ImGui::Checkbox("Frustum Culling", reinterpret_cast<bool *>(&Renderer::instance()->Culling.frustum_culling));
		ImGui::Checkbox("Back Face Cone Culling", reinterpret_cast<bool *>(&Renderer::instance()->Culling.backface_culling));
		ImGui::Checkbox("Hi-z Occlusion Culling", reinterpret_cast<bool *>(&Renderer::instance()->Culling.occulsion_culling));
	});

	draw_node("Color Correction", []() {
		ImGui::DragFloat("Exposure", &Renderer::instance()->Color_Correction.exposure, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.2f");
		ImGui::DragFloat("Gamma", &Renderer::instance()->Color_Correction.gamma, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.2f");
	});

	draw_node("Bloom", []() {
		ImGui::Checkbox("Enable", reinterpret_cast<bool *>(&Renderer::instance()->Bloom.enable));
		ImGui::DragFloat("Threshold", &Renderer::instance()->Bloom.threshold, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.3f");
		ImGui::DragFloat("Scale", &Renderer::instance()->Bloom.scale, 0.001f, 0.f, std::numeric_limits<float>::max(), "%.3f");
		ImGui::DragFloat("Strength", &Renderer::instance()->Bloom.strength, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.3f");
	});

	draw_node("Temporal Anti Alias", []() {
		ImGui::Checkbox("Enable", reinterpret_cast<bool *>(&Renderer::instance()->TAA.enable));
		ImGui::SliderFloat("Feedback Min", &Renderer::instance()->TAA.feedback.x, 0.f, 1.f, "%.3f");
		ImGui::SliderFloat("Feedback Max", &Renderer::instance()->TAA.feedback.y, 0.f, 1.f, "%.3f");
	});

	draw_node("Environment Light", []() {
		const char *const environment_light_type[] = {"None", "HDRI"};
		int               current                  = static_cast<int>(Renderer::instance()->EnvLight.type);
		ImGui::Combo("Environment Light", &current, environment_light_type, 2);
		Renderer::instance()->EnvLight.type = static_cast<Renderer::EnvLightType>(current);

		if (current == 1 || current == 2)
		{
			ImGui::PushID("Environment Light");
			if (ImGui::ImageButton(Renderer::instance()->getResourceCache().hasImage(FileSystem::getRelativePath(Renderer::instance()->EnvLight.filename)) ?
                                       ImGuiContext::textureID(Renderer::instance()->getResourceCache().loadImage(FileSystem::getRelativePath(Renderer::instance()->EnvLight.filename)), Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp)) :
                                       ImGuiContext::textureID(Renderer::instance()->getDefaultTexture(), Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp)),
			                       ImVec2{100.f, 100.f}))
			{
				Renderer::instance()->EnvLight.filename = "";
				Renderer::instance()->EnvLight.update   = true;
			}
			ImGui::PopID();

			if (ImGui::BeginDragDropTarget())
			{
				if (const auto *pay_load = ImGui::AcceptDragDropPayload("Texture2D"))
				{
					ASSERT(pay_load->DataSize == sizeof(std::string));
					if (Renderer::instance()->EnvLight.filename != *static_cast<std::string *>(pay_load->Data))
					{
						Renderer::instance()->EnvLight.filename = *static_cast<std::string *>(pay_load->Data);
						Renderer::instance()->EnvLight.update   = true;
					}
				}
				ImGui::EndDragDropTarget();
			}
		}
	});

	ImGui::End();
}
}        // namespace Ilum::panel