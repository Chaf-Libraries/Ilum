#include "RenderSetting.hpp"

#include "Graphics/GraphicsContext.hpp"

#include "Renderer/RenderGraph/RenderGraph.hpp"
#include "Renderer/Renderer.hpp"

#include "ImGui/ImGuiContext.hpp"

#include "File/FileSystem.hpp"

#include "Timing/Timer.hpp"

#include <imgui.h>
#include <imgui_internal.h>

namespace Ilum::panel
{
template <typename Callback>
inline void draw_node(const std::string &name, Callback callback)
{
	const ImGuiTreeNodeFlags tree_node_flags          = ImGuiTreeNodeFlags_Framed | ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_AllowItemOverlap | ImGuiTreeNodeFlags_FramePadding;
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
	m_name = "Renderer Inspector";
	m_stopwatch.start();
}

void RenderSetting::draw(float delta_time)
{
	ImGui::Begin("Renderer Inspector", &active);

	const char *const render_mode[]       = {"Polygon", "Wire Frame", "Point Cloud"};
	int               current_render_mode = static_cast<int>(Renderer::instance()->Render_Mode);

	ImGui::PushItemWidth(100.f);
	if (ImGui::Combo("Render Mode", &current_render_mode, render_mode, 3))
	{
		Renderer::instance()->Render_Mode = static_cast<Renderer::RenderMode>(current_render_mode);
		Renderer::instance()->update();
	}
	ImGui::PopItemWidth();

	ImGui::SameLine();

	bool vsync = GraphicsContext::instance()->isVsync();

	if (ImGui::Checkbox("V-Sync", &vsync))
	{
		GraphicsContext::instance()->setVsync(vsync);
	}

	auto *render_graph = Renderer::instance()->getRenderGraph();
	render_graph->onImGui();

	draw_node("Render Pass", [&render_graph]() {
		for (auto &render_node : render_graph->getNodes())
		{
			draw_node(render_node.name, [&render_node]() {
				render_node.pass->onImGui();
			});
		}
	});

	draw_node("Culling", []() {
		ImGui::Checkbox("Frustum Culling", reinterpret_cast<bool *>(&Renderer::instance()->Culling.frustum_culling));
		ImGui::Checkbox("Back Face Cone Culling", reinterpret_cast<bool *>(&Renderer::instance()->Culling.backface_culling));
		ImGui::Checkbox("Hi-z Occlusion Culling", reinterpret_cast<bool *>(&Renderer::instance()->Culling.occulsion_culling));
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

	draw_node("Profiler", [render_graph, this]() {
		if (m_stopwatch.elapsedSecond() > 0.1f)
		{
			if (Timer::instance()->getFPS() != 0.0)
			{
				if (m_frame_times.size() < 50)
				{
					m_frame_times.push_back(1000.0f / static_cast<float>(Timer::instance()->getFPS()));
				}
				else
				{
					std::rotate(m_frame_times.begin(), m_frame_times.begin() + 1, m_frame_times.end());
					m_frame_times.back() = 1000.0f / static_cast<float>(Timer::instance()->getFPS());
				}
			}

			m_stopwatch.start();
		}

		float min_frame_time = 0.f, max_frame_time = 0.f;
		float max_cpu_time = 0.f, max_gpu_time = 0.f;

		if (!m_frame_times.empty())
		{
			min_frame_time = *std::min_element(m_frame_times.begin(), m_frame_times.end());
			max_frame_time = *std::max_element(m_frame_times.begin(), m_frame_times.end());
		}

		std::vector<float> cpu_times;
		std::vector<float> gpu_times;

		if (ImGui::BeginTable("CPU&GPU Time", 3, ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders))
		{
			ImGui::TableSetupColumn("Pass");
			ImGui::TableSetupColumn("CPU Time (ms)");
			ImGui::TableSetupColumn("GPU Time (ms)");
			ImGui::TableHeadersRow();

			for (auto &render_node : render_graph->getNodes())
			{
				cpu_times.push_back(render_node.pass->getCPUTime());
				gpu_times.push_back(render_node.pass->getGPUTime());

				ImGui::TableNextRow();

				ImGui::TableSetColumnIndex(0);
				ImGui::Text("%s", render_node.name.c_str());
				ImGui::TableSetColumnIndex(1);
				ImGui::Text("%f", cpu_times.back());
				ImGui::TableSetColumnIndex(2);
				ImGui::Text("%f", gpu_times.back());
			}
			ImGui::EndTable();
		}

		if (!cpu_times.empty())
		{
			max_cpu_time = *std::max_element(cpu_times.begin(), cpu_times.end());
			max_gpu_time = *std::max_element(gpu_times.begin(), gpu_times.end());
		}

		ImGui::PlotLines(("Frame Times (" + std::to_string(static_cast<uint32_t>(Timer::instance()->getFPS())) + "fps)").c_str(), m_frame_times.data(), static_cast<int>(m_frame_times.size()), 0, nullptr, min_frame_time * 0.8f, max_frame_time * 1.2f, ImVec2{0, 80});
		ImGui::PlotHistogram("CPU Times", cpu_times.data(), static_cast<int>(cpu_times.size()), 0, nullptr, 0.f, max_cpu_time * 1.2f, ImVec2(0, 80.0f));
		ImGui::PlotHistogram("GPU Times", gpu_times.data(), static_cast<int>(gpu_times.size()), 0, nullptr, 0.f, max_gpu_time * 1.2f, ImVec2(0, 80.0f));
	});

	ImGui::End();
}
}        // namespace Ilum::panel