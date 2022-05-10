#pragma once

#include "MeshRenderer.hpp"

#include <Asset/AssetManager.hpp>

namespace Ilum::cmpt
{
bool MeshRenderer::OnImGui(ImGuiContext &context)
{
	if (mesh && manager && manager->IsValid(mesh))
	{
		return mesh->OnImGui(context);
	}
	return false;
}

}        // namespace Ilum::cmpt