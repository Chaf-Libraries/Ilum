#pragma once

#include "Component.hpp"

#include <Asset/Mesh.hpp>

namespace Ilum::cmpt
{
struct MeshRenderer : public Component
{
	MeshRenderer() = default;

	Mesh *mesh = nullptr;
	AssetManager *manager = nullptr;
	

	template <class Archive>
	void serialize(Archive &ar)
	{
		// ar();
	}

	bool OnImGui(ImGuiContext &context) override;
};
}        // namespace Ilum::cmpt