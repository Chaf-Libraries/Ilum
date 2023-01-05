#pragma once

#include <Editor/Editor.hpp>
#include <Material/MaterialCompiler.hpp>
#include <Material/MaterialGraph.hpp>
#include <Material/MaterialNode.hpp>
#include <Renderer/Renderer.hpp>
#include <Resource/ResourceManager.hpp>

#include <imgui.h>

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
