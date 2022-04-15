#include "RendererInspector.hpp"

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

RendererInspector::RendererInspector()
{
	m_name = "Renderer Inspector";
	m_stopwatch.start();
}

void RendererInspector::draw(float delta_time)
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
			if (ImGui::TreeNode(render_node.name.c_str()))
			{
				render_node.pass->onImGui();
				ImGui::TreePop();
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

		if (ImGui::BeginTable("CPU&GPU Time", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders))
		{
			ImGui::TableSetupColumn("Pass");
			ImGui::TableSetupColumn("CPU Time (ms)");
			ImGui::TableSetupColumn("GPU Time (ms)");
			ImGui::TableSetupColumn("Record Thread");
			ImGui::TableHeadersRow();

			for (size_t i = 0; i < render_graph->getNodes().size(); i++)
			{
				const auto &render_node = render_graph->getNodes()[i];

				cpu_times.push_back(render_node.pass->getCPUTime());
				gpu_times.push_back(render_node.pass->getGPUTime());

				ImGui::TableNextRow();

				ImGui::TableSetColumnIndex(0);
				ImGui::Text("%s", (std::to_string(i) + " - " + render_node.name).c_str());
				ImGui::TableSetColumnIndex(1);
				ImGui::Text("%f", cpu_times.back());
				ImGui::TableSetColumnIndex(2);
				ImGui::Text("%f", gpu_times.back());
				ImGui::TableSetColumnIndex(3);
				ImGui::Text("%zu", render_node.pass->getThreadID());
			}
			ImGui::EndTable();
		}

		if (!cpu_times.empty())
		{
			max_cpu_time = *std::max_element(cpu_times.begin(), cpu_times.end());
			max_gpu_time = *std::max_element(gpu_times.begin(), gpu_times.end());
		}
		ImGui::Text("Render Target Resolution: %d x %d", Renderer::instance()->getRenderTargetExtent().width, Renderer::instance()->getRenderTargetExtent().height);
		ImGui::Text("Viewport Size: %d x %d", Renderer::instance()->getViewportExtent().width, Renderer::instance()->getViewportExtent().height);
		ImGui::PlotLines(("Frame Times (" + std::to_string(static_cast<uint32_t>(Timer::instance()->getFPS())) + "fps)").c_str(), m_frame_times.data(), static_cast<int>(m_frame_times.size()), 0, nullptr, min_frame_time * 0.8f, max_frame_time * 1.2f, ImVec2{0, 80});
		ImGui::PlotHistogram("CPU Times", cpu_times.data(), static_cast<int>(cpu_times.size()), 0, nullptr, 0.f, max_cpu_time * 1.2f, ImVec2(0, 80.0f));
		ImGui::PlotHistogram("GPU Times", gpu_times.data(), static_cast<int>(gpu_times.size()), 0, nullptr, 0.f, max_gpu_time * 1.2f, ImVec2(0, 80.0f));

		if (ImGui::TreeNode("Static Mesh Stats"))
		{
			ImGui::Text("Instance Visibility:");
			ImGui::SameLine();
			ImGui::ProgressBar(static_cast<float>(Renderer::instance()->Render_Stats.static_mesh_count.instance_visible) / static_cast<float>(Renderer::instance()->Render_Stats.static_mesh_count.instance_count),
			                   ImVec2(0.f, 0.f),
			                   (std::to_string(Renderer::instance()->Render_Stats.static_mesh_count.instance_visible) + "/" + std::to_string(Renderer::instance()->Render_Stats.static_mesh_count.instance_count)).c_str());

			ImGui::Text("Meshlet Visibility:");
			ImGui::SameLine();
			ImGui::ProgressBar(static_cast<float>(Renderer::instance()->Render_Stats.static_mesh_count.meshlet_visible) / static_cast<float>(Renderer::instance()->Render_Stats.static_mesh_count.meshlet_count),
			                   ImVec2(0.f, 0.f),
			                   (std::to_string(Renderer::instance()->Render_Stats.static_mesh_count.meshlet_visible) + "/" + std::to_string(Renderer::instance()->Render_Stats.static_mesh_count.meshlet_count)).c_str());

			ImGui::Text("Triangle Count: %d", Renderer::instance()->Render_Stats.static_mesh_count.triangle_count);
			ImGui::TreePop();
		}

		if (ImGui::TreeNode("Dynamic Mesh Stats"))
		{
			ImGui::Text("Instance Count: %d", Renderer::instance()->Render_Stats.dynamic_mesh_count.instance_count);
			ImGui::Text("Triangle Count: %d", Renderer::instance()->Render_Stats.dynamic_mesh_count.triangle_count);
			ImGui::TreePop();
		}

		if (ImGui::TreeNode("Light Stats"))
		{
			ImGui::Text("Directional Light Count: %d", Renderer::instance()->Render_Stats.light_count.directional_light_count);
			ImGui::Text("Point Light Count: %d", Renderer::instance()->Render_Stats.light_count.point_light_count);
			ImGui::Text("Spot Light Count: %d", Renderer::instance()->Render_Stats.light_count.spot_light_count);
			ImGui::TreePop();
		}
	});

	ImGui::End();
}
}        // namespace Ilum::panel