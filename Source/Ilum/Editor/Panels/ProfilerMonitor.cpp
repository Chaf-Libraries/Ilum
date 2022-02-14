#include "ProfilerMonitor.hpp"

#include "Graphics/GraphicsContext.hpp"
#include "Graphics/Profiler.hpp"

#include "Renderer/Renderer.hpp"

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

ProfilerMonitor::ProfilerMonitor()
{
	m_name = "Profiler";
	m_timer.Tick();
}

void ProfilerMonitor::draw(float delta_time)
{
	ImGui::Begin("Profiler", &active);

	float fps = 1.f / ImGui::GetIO().DeltaTime;

	if (m_timer.Elapsed() > 200.f)
	{
		m_profile_result = GraphicsContext::instance()->getProfiler().getResult();

		if (fps != 0.0)
		{
			if (m_frame_times.size() < 50)
			{
				m_frame_times.push_back(1000.0f / static_cast<float>(fps));
			}
			else
			{
				std::rotate(m_frame_times.begin(), m_frame_times.begin() + 1, m_frame_times.end());
				m_frame_times.back() = 1000.0f / static_cast<float>(fps);
			}
		}

		m_timer.Tick();
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
	uint32_t           index = 0;

	if (ImGui::BeginTable("CPU&GPU Time", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders))
	{
		ImGui::PushItemWidth(5.f);
		ImGui::TableSetupColumn("Index");
		ImGui::PopItemWidth();
		ImGui::TableSetupColumn("Pass");
		ImGui::TableSetupColumn("CPU Time (ms)");
		ImGui::TableSetupColumn("GPU Time (ms)");
		ImGui::TableHeadersRow();

		for (auto &[name, res] : m_profile_result)
		{
			auto [cpu_time, gpu_time] = res;

			cpu_times.push_back(cpu_time);
			gpu_times.push_back(gpu_time);

			ImGui::TableNextRow();

			ImGui::TableSetColumnIndex(0);

			ImGui::Text("%d", index++);

			ImGui::TableSetColumnIndex(1);
			ImGui::Text("%s", name.c_str());
			ImGui::TableSetColumnIndex(2);
			ImGui::Text("%f", cpu_time);
			ImGui::TableSetColumnIndex(3);
			ImGui::Text("%f", gpu_time);
		}

		ImGui::EndTable();
	}

	if (!m_profile_result.empty())
	{
		if (!cpu_times.empty())
		{
			max_cpu_time = *std::max_element(cpu_times.begin(), cpu_times.end());
		}
		if (!gpu_times.empty())
		{
			max_gpu_time = *std::max_element(gpu_times.begin(), gpu_times.end());
		}
	}

	ImGui::PlotLines(("Frame Times (" + std::to_string(static_cast<uint32_t>(1.f / ImGui::GetIO().DeltaTime)) + "fps)").c_str(), m_frame_times.data(), static_cast<int>(m_frame_times.size()), 0, nullptr, min_frame_time * 0.8f, max_frame_time * 1.2f, ImVec2{0, 80});
	ImGui::PlotHistogram("CPU Times", cpu_times.data(), static_cast<int>(cpu_times.size()), 0, nullptr, 0.f, max_cpu_time * 1.2f, ImVec2(0, 80.0f));
	ImGui::PlotHistogram("GPU Times", gpu_times.data(), static_cast<int>(gpu_times.size()), 0, nullptr, 0.f, max_gpu_time * 1.2f, ImVec2(0, 80.0f));

	draw_node("Static Mesh Stats", []() {
		ImGui::Text("Instance Count: %d", Renderer::instance()->Render_Stats.static_mesh_count.instance_count);
		ImGui::Text("Instance Visible: %d", Renderer::instance()->Render_Stats.static_mesh_count.instance_visible);
		ImGui::Text("Meshlet Count: %d", Renderer::instance()->Render_Stats.static_mesh_count.meshlet_count);
		ImGui::Text("Meshlet Visible: %d", Renderer::instance()->Render_Stats.static_mesh_count.meshlet_visible);
		ImGui::Text("Triangle Count: %d", Renderer::instance()->Render_Stats.static_mesh_count.triangle_count);
	});

	draw_node("Dynamic Mesh Stats", []() {
		ImGui::Text("Instance Count: %d", Renderer::instance()->Render_Stats.dynamic_mesh_count.instance_count);
		ImGui::Text("Triangle Count: %d", Renderer::instance()->Render_Stats.dynamic_mesh_count.triangle_count);
	});

	draw_node("Light Stats", []() {
		ImGui::Text("Directional Light Count: %d", Renderer::instance()->Render_Stats.light_count.directional_light_count);
		ImGui::Text("Point Light Count: %d", Renderer::instance()->Render_Stats.light_count.point_light_count);
		ImGui::Text("Spot Light Count: %d", Renderer::instance()->Render_Stats.light_count.spot_light_count);
	});

	ImGui::End();
}
}        // namespace Ilum::panel