#include <Animation/Animation.hpp>
#include <Editor/Widget.hpp>
#include <Editor/Editor.hpp>

#include <imgui.h>
#include <imgui_internal.h>

using namespace Ilum;

class AnimationEditor : public Widget
{
  public:
	AnimationEditor(Editor *editor) :
	    Widget("Animation Editor", editor)
	{
	}

	virtual ~AnimationEditor() = default;

	virtual void Tick() override
	{
		if (!ImGui::Begin(m_name.c_str()))
		{
			ImGui::End();
			return;
		}

		ImGui::End();
	}
};

extern "C"
{
	EXPORT_API AnimationEditor *Create(Editor *editor, ImGuiContext *context)
	{
		ImGui::SetCurrentContext(context);
		return new AnimationEditor(editor);
	}
}