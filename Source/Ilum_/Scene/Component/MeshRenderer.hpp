#pragma once

#include "Component.hpp"

#include <Asset/Mesh.hpp>

namespace Ilum::cmpt
{
class MeshRenderer : public Component
{
  public:
	MeshRenderer() = default;

	void SetMesh(Mesh *mesh);

	Buffer *GetBuffer();
	Mesh *GetMesh();

	virtual void Tick(Scene &scene, entt::entity entity, RHIDevice *device) override;

	bool OnImGui(ImGuiContext &context) override;

  private:
	Mesh	     *m_mesh    = nullptr;
	AssetManager *m_manager = nullptr;

	std::unique_ptr<Buffer> m_buffer = nullptr;
};
}        // namespace Ilum::cmpt