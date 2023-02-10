#include "RenderGraphInspector.hpp"
#include "Editor/Editor.hpp"
#include "Editor/ImGui/ImGuiHelper.hpp"

#include <Core/Time.hpp>
#include <RenderGraph/RenderGraph.hpp>
#include <Renderer/Renderer.hpp>

#include <imgui.h>

namespace Ilum
{
RenderGraphInspector::RenderGraphInspector(Editor *editor) :
    Widget("Render Graph Inspector", editor)
{
}

RenderGraphInspector::~RenderGraphInspector()
{
}

void RenderGraphInspector::Tick()
{
	ImGui::Begin(m_name.c_str());

	auto *renderer     = p_editor->GetRenderer();
	auto *render_graph = renderer->GetRenderGraph();
	if (render_graph)
	{
		const auto &render_passes = render_graph->GetRenderPasses();
		for (const auto &pass : render_passes)
		{
			ImGui::PushID(&pass.config);
			if (ImGui::TreeNode(pass.name.c_str()))
			{
				ImGui::EditVariant(pass.name, p_editor, pass.config);
				ImGui::TreePop();
			}
			ImGui::PopID();
		}

		ImGui::Separator();

		{
			static float t = 0;
			t += ImGui::GetIO().DeltaTime;
			if (t > 0.1f)
			{
				t = 0;
				if (ImGui::GetIO().Framerate != 0.0)
				{
					if (m_frame_times.size() < 50)
					{
						m_frame_times.push_back(1000.0f / ImGui::GetIO().Framerate);
					}
					else
					{
						std::rotate(m_frame_times.begin(), m_frame_times.begin() + 1, m_frame_times.end());
						m_frame_times.back() = 1000.0f / ImGui::GetIO().Framerate;
					}
				}
			}

			float min_frame_time = 0.f, max_frame_time = 0.f;
			float max_cpu_time = 0.f, max_gpu_time = 0.f;

			if (!m_frame_times.empty())
			{
				min_frame_time = *std::min_element(m_frame_times.begin(), m_frame_times.end());
				max_frame_time = *std::max_element(m_frame_times.begin(), m_frame_times.end());
			}

			std::vector<float>        cpu_times;
			std::vector<float>        gpu_times;
			std::vector<const char *> pass_names;

			if (ImGui::BeginTable("CPU&GPU Time", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders))
			{
				ImGui::TableSetupColumn("Pass");
				ImGui::TableSetupColumn("CPU Time (ms)");
				ImGui::TableSetupColumn("GPU Time (ms)");
				ImGui::TableSetupColumn("Thread ID");
				ImGui::TableHeadersRow();

				uint32_t idx = 0;
				for (const auto &pass : render_passes)
				{
					const auto &profiler_state = pass.profiler->GetProfileState();

					cpu_times.push_back(profiler_state.cpu_time);
					gpu_times.push_back(profiler_state.gpu_time);
					pass_names.push_back(pass.name.c_str());

					ImGui::TableNextRow();

					ImGui::TableSetColumnIndex(0);
					ImGui::Text("%s", (std::to_string(idx++) + " - " + pass.name).c_str());
					ImGui::TableSetColumnIndex(1);
					ImGui::Text("%f", cpu_times.back());
					ImGui::TableSetColumnIndex(2);
					ImGui::Text("%f", gpu_times.back());
					ImGui::TableSetColumnIndex(3);
					ImGui::Text("%zu", profiler_state.thread_id);
				}
				ImGui::EndTable();
			}

			if (!cpu_times.empty())
			{
				max_cpu_time = *std::max_element(cpu_times.begin(), cpu_times.end());
				max_gpu_time = *std::max_element(gpu_times.begin(), gpu_times.end());
			}

			ImGui::PlotLines(("Frame Times (" + std::to_string(static_cast<uint32_t>(ImGui::GetIO().Framerate)) + "fps)").c_str(), m_frame_times.data(), static_cast<int>(m_frame_times.size()), 0, nullptr, min_frame_time * 0.8f, max_frame_time * 1.2f, ImVec2{0, 80});
			ImGui::PlotHistogram("CPU Times", cpu_times.data(), static_cast<int>(cpu_times.size()), 0, nullptr, 0.f, max_cpu_time * 1.2f, ImVec2(0, 80.0f));
			ImGui::PlotHistogram("GPU Times", gpu_times.data(), static_cast<int>(gpu_times.size()), 0, nullptr, 0.f, max_gpu_time * 1.2f, ImVec2(0, 80.0f));
		}
	}

	ImGui::End();
}
}        // namespace Ilum