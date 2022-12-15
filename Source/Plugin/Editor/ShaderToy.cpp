#include <Editor/Editor.hpp>
#include <Editor/Widget.hpp>
#include <Renderer/Renderer.hpp>

#include <imgui.h>
#include <imgui_internal.h>

using namespace Ilum;

class ShaderToy : public Widget
{
  public:
	ShaderToy(Editor *editor) :
	    Widget("Shader Toy", editor)
	{
	}

	virtual ~ShaderToy() override = default;

	virtual void Tick() override
	{
		if (!ImGui::Begin(m_name.c_str(), nullptr, ImGuiWindowFlags_MenuBar))
		{
			ImGui::End();
			return;
		}

		if (ImGui::BeginMenuBar())
		{
			if (ImGui::MenuItem("Load"))
			{

			}

			ImGui::EndMenuBar();
		}


		ImGui::End();
	}
};

extern "C"
{
	EXPORT_API ShaderToy *Create(Editor *editor, ImGuiContext *context)
	{
		ImGui::SetCurrentContext(context);
		return new ShaderToy(editor);
	}
}