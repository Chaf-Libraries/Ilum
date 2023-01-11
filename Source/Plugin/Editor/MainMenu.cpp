#include <Scene/Components/AllComponents.hpp>
#include <Editor/Editor.hpp>
#include <Editor/Widget.hpp>
#include <Renderer/Renderer.hpp>
#include <Scene/Node.hpp>
#include <Scene/Scene.hpp>

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
					char *path = nullptr;
					if (NFD_OpenDialog("scene", Path::GetInstance().GetCurrent(false).c_str(), &path) == NFD_OKAY)
					{
						std::ifstream is(path, std::ios::binary);
						InputArchive archive(is);
						p_editor->GetRenderer()->GetScene()->Load(archive);
					}
				}

				if (ImGui::MenuItem("Save Scene"))
				{
					char *path = nullptr;
					if (NFD_SaveDialog("scene", Path::GetInstance().GetCurrent(false).c_str(), &path) == NFD_OKAY)
					{
						std::ofstream os(Path::GetInstance().GetFileExtension(path) == ".scene" ? path : std::string(path) + ".scene", std::ios::binary);
						OutputArchive archive(os);
						p_editor->GetRenderer()->GetScene()->Save(archive);
					}
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
						auto *node = p_editor->GetRenderer()->GetScene()->CreateNode("Spot Light");
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

					if (ImGui::MenuItem("Rectangle Light"))
					{
						auto *node = p_editor->GetRenderer()->GetScene()->CreateNode("Polygonal Light");
						node->AddComponent(std::make_unique<Cmpt::Transform>(node));
						node->AddComponent(std::make_unique<Cmpt::RectLight>(node));
					}

					ImGui::EndMenu();
				}

				if (ImGui::BeginMenu("Add Camera"))
				{
					if (ImGui::MenuItem("Perspective Camera"))
					{
						auto *node = p_editor->GetRenderer()->GetScene()->CreateNode("Perspective Camera");
						node->AddComponent(std::make_unique<Cmpt::Transform>(node));
						auto *camera = node->AddComponent(std::make_unique<Cmpt::PerspectiveCamera>(node));

						if (!p_editor->GetMainCamera())
						{
							p_editor->SetMainCamera(camera);
						}
					}

					if (ImGui::MenuItem("Orthographic Camera"))
					{
						auto *node = p_editor->GetRenderer()->GetScene()->CreateNode("Orthographic Camera");
						node->AddComponent(std::make_unique<Cmpt::Transform>(node));
						node->AddComponent(std::make_unique<Cmpt::OrthographicCamera>(node));
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