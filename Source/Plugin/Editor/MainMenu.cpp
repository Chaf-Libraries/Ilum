#include <Components/AllComponents.hpp>
#include <Editor/Editor.hpp>
#include <Editor/Widget.hpp>
#include <Renderer/Renderer.hpp>
#include <SceneGraph/Node.hpp>
#include <SceneGraph/Scene.hpp>

#include <imgui.h>
#include <imgui_internal.h>

#include <nfd.h>

using namespace Ilum;

class MainMenu : public Widget
{
  public:
	MainMenu(Editor *editor) :
	    Widget("Main Menu", editor)
	{
	}

	virtual ~MainMenu() override = default;

	virtual void Tick() override
	{
		if (ImGui::BeginMainMenuBar())
		{
			if (ImGui::BeginMenu("File"))
			{
				if (ImGui::MenuItem("Load Scene"))
				{
				}

				if (ImGui::MenuItem("Save Scene"))
				{
				}

				if (ImGui::MenuItem("Import Model"))
				{

				}

				ImGui::EndMenu();
			}

			if (ImGui::BeginMenu("Component"))
			{
				if (ImGui::BeginMenu("Add Light"))
				{
					if (ImGui::MenuItem("Spot Light"))
					{
						auto* node = p_editor->GetRenderer()->GetScene()->CreateNode("Spot Light");
						node->AddComponent(std::make_unique<Cmpt::Transform>(node));
						node->AddComponent(std::make_unique<Cmpt::SpotLight>(node));
					}

					if (ImGui::MenuItem("Point Light"))
					{
						auto *node = p_editor->GetRenderer()->GetScene()->CreateNode("Point Light");
						node->AddComponent(std::make_unique<Cmpt::Transform>(node));
						node->AddComponent(std::make_unique<Cmpt::PointLight>(node));
					}

					if (ImGui::MenuItem("Directional Light"))
					{
						auto *node = p_editor->GetRenderer()->GetScene()->CreateNode("Directional Light");
						node->AddComponent(std::make_unique<Cmpt::Transform>(node));
						node->AddComponent(std::make_unique<Cmpt::DirectionalLight>(node));
					}

					if (ImGui::MenuItem("Polygonal Light"))
					{
						auto *node = p_editor->GetRenderer()->GetScene()->CreateNode("Polygonal Light");
						node->AddComponent(std::make_unique<Cmpt::Transform>(node));
						node->AddComponent(std::make_unique<Cmpt::PolygonLight>(node));
					}

					ImGui::EndMenu();
				}

				if (ImGui::BeginMenu("Add Camera"))
				{
					if (ImGui::MenuItem("Perspective Camera"))
					{
					}

					if (ImGui::MenuItem("Orthographic Camera"))
					{
					}

					ImGui::EndMenu();
				}

				ImGui::EndMenu();
			}

			ImGui::EndMainMenuBar();
		}
	}
};

extern "C"
{
	EXPORT_API MainMenu *Create(Editor *editor, ImGuiContext *context)
	{
		ImGui::SetCurrentContext(context);
		Ilum::Cmpt::SetImGuiContext(context);
		return new MainMenu(editor);
	}
}