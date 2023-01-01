#pragma once

#include "Material/MaterialNode.hpp"

#include <imgui.h>

#define CONFIGURATION_MATERIAL_NODE(NODE)                                      \
	extern "C"                                                                 \
	{                                                                          \
		EXPORT_API void Create(MaterialNodeDesc *desc, size_t *handle)         \
		{                                                                      \
			*desc = NODE::GetInstance().Create(*handle);                       \
		}                                                                      \
		EXPORT_API void OnImGui(MaterialNodeDesc *desc, ImGuiContext *context) \
		{                                                                      \
			ImGui::SetCurrentContext(context);                                 \
			NODE::GetInstance().OnImGui(*desc);                                \
		}                                                                      \
	}
