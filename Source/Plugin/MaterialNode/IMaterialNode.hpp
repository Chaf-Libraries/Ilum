#pragma once

#include <Editor/Editor.hpp>
#include <Material/MaterialCompiler.hpp>
#include <Material/MaterialGraph.hpp>
#include <Material/MaterialNode.hpp>
#include <Renderer/Renderer.hpp>
#include <Resource/ResourceManager.hpp>

#include <imgui.h>

using namespace Ilum;

template <typename _Ty>
class MaterialNode
{
  public:
	static _Ty &GetInstance()
	{
		static _Ty node;
		return node;
	}

	virtual MaterialNodeDesc Create(size_t &handle) = 0;

	virtual void OnImGui(MaterialNodeDesc &node_desc, Editor *editor) = 0;

	virtual void EmitHLSL(const MaterialNodeDesc &node_desc, const MaterialGraphDesc &graph_desc, ResourceManager *manager, MaterialCompilationContext *context) = 0;
};

#define CONFIGURATION_MATERIAL_NODE(NODE)                                                                                                                         \
	extern "C"                                                                                                                                                    \
	{                                                                                                                                                             \
		EXPORT_API void Create(MaterialNodeDesc *desc, size_t *handle)                                                                                            \
		{                                                                                                                                                         \
			*desc = NODE::GetInstance().Create(*handle);                                                                                                          \
		}                                                                                                                                                         \
		EXPORT_API void OnImGui(MaterialNodeDesc *desc, Editor *editor, ImGuiContext *context)                                                                    \
		{                                                                                                                                                         \
			ImGui::SetCurrentContext(context);                                                                                                                    \
			NODE::GetInstance().OnImGui(*desc, editor);                                                                                                           \
		}                                                                                                                                                         \
		EXPORT_API void EmitHLSL(const MaterialNodeDesc &node_desc, const MaterialGraphDesc &graph_desc, ResourceManager *manager, MaterialCompilationContext *context) \
		{                                                                                                                                                         \
			NODE::GetInstance().EmitHLSL(node_desc, graph_desc, manager, context);                                                                               \
		}                                                                                                                                                         \
	}
