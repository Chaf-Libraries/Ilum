#include <Editor/Editor.hpp>
#include <Editor/Widget.hpp>
#include <RHI/RHIContext.hpp>
#include <RenderGraph/RenderGraph.hpp>
#include <Renderer/Renderer.hpp>
#include <Scene/Components/AllComponents.hpp>
#include <Scene/Node.hpp>

#include <imgui.h>
#include <imgui_internal.h>

using namespace Ilum;

class Inspector : public Widget
{
  public:
	Inspector(Editor *editor) :
	    Widget("Inspector", editor)
	{
	}

	virtual ~Inspector() override = default;

	virtual void Tick() override
	{
		auto *select = p_editor->GetSelectedNode();

		if (!ImGui::Begin(m_name.c_str()))
		{
			ImGui::End();
			return;
		}

		// Draw node components
		if (select)
		{
			DrawNodeInspector(select);
		}
		else
		{
			DrawRendererInspector();
		}

		ImGui::End();
	}

	void DrawNodeInspector(Node *node)
	{
		{
			// Name
			char buf[64] = {0};
			std::memcpy(buf, node->GetName().data(), sizeof(buf));
			ImGui::PushItemWidth(150.f);
			if (ImGui::InputText("Tag", buf, sizeof(buf)))
			{
				node->SetName(buf);
			}
			ImGui::PopItemWidth();
		}

		for (auto &[type, cmpt] : node->GetComponents())
		{
			const ImGuiTreeNodeFlags tree_node_flags = ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_Framed | ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_AllowItemOverlap | ImGuiTreeNodeFlags_FramePadding;

			ImVec2 content_region_available = ImGui::GetContentRegionAvail();

			ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2{4, 4});
			float line_height = GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f;
			bool  open        = ImGui::TreeNodeEx((void *) type.hash_code(), tree_node_flags, cmpt->GetName());
			ImGui::PopStyleVar();

			bool update = false;

			if (open)
			{
				cmpt->OnImGui();
				ImGui::TreePop();
			}
		}
	}

	void DrawRendererInspector()
	{
		const ImGuiTreeNodeFlags tree_node_flags = ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_Framed | ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_AllowItemOverlap | ImGuiTreeNodeFlags_FramePadding;

		auto *rhi_context  = p_editor->GetRHIContext();
		auto *renderer     = p_editor->GetRenderer();
		auto *render_graph = renderer->GetRenderGraph();

		if (ImGui::TreeNodeEx("Hardware Info", tree_node_flags, "Hardware Info"))
		{
			ImGui::Text("GPU: %s", rhi_context->GetDeviceName().c_str());
			ImGui::Text("Backend: %s", rhi_context->GetBackend().c_str());
			ImGui::Text("CUDA: %s", rhi_context->HasCUDA() ? "Enable" : "Disable");
			ImGui::TreePop();
		}

		if (render_graph && ImGui::TreeNodeEx("Profiler", tree_node_flags, "Profiler"))
		{
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
					ImGui::TableSetupColumn("CPU (ms)");
					ImGui::TableSetupColumn("GPU (ms)");
					ImGui::TableSetupColumn("Thread");
					ImGui::TableHeadersRow();

					uint32_t idx = 0;
					for (const auto &pass : render_graph->GetRenderPasses())
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

				ImGui::PlotLines(("Frame Time (" + std::to_string(static_cast<uint32_t>(ImGui::GetIO().Framerate)) + "fps)").c_str(), m_frame_times.data(), static_cast<int>(m_frame_times.size()), 0, nullptr, min_frame_time * 0.8f, max_frame_time * 1.2f, ImVec2{0, 80});
				ImGui::PlotHistogram("CPU Time", cpu_times.data(), static_cast<int>(cpu_times.size()), 0, nullptr, 0.f, max_cpu_time * 1.2f, ImVec2(0, 80.0f));
				ImGui::PlotHistogram("GPU Time", gpu_times.data(), static_cast<int>(gpu_times.size()), 0, nullptr, 0.f, max_gpu_time * 1.2f, ImVec2(0, 80.0f));
			}

			ImGui::TreePop();
		}

		if (render_graph && ImGui::TreeNodeEx("Render Info", tree_node_flags, "Render Info"))
		{
			for (const auto &pass : render_graph->GetRenderPasses())
			{
				ImGui::PushID(&pass.config);
				if (!pass.config.Empty() && ImGui::TreeNodeEx(&pass, tree_node_flags, pass.name.c_str()))
				{
					PluginManager::GetInstance().Call<bool>(fmt::format("shared/RenderPass/RenderPass.{}.{}.dll", pass.category, pass.name), "OnImGui", &pass.config, ImGui::GetCurrentContext());
					ImGui::TreePop();
				}
				ImGui::PopID();
			}
			ImGui::TreePop();
		}
	}

  private:
	std::vector<float> m_frame_times;
};

extern "C"
{
	EXPORT_API Inspector *Create(Editor *editor, ImGuiContext *context)
	{
		ImGui::SetCurrentContext(context);
		Ilum::Cmpt::SetImGuiContext(context);
		return new Inspector(editor);
	}
}