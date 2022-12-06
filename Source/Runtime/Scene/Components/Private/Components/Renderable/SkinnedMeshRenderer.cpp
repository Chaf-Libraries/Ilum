#include "Renderable/SkinnedMeshRenderer.hpp"
#include <SceneGraph/Node.hpp>

namespace Ilum
{
namespace Cmpt
{
SkinnedMeshRenderer::SkinnedMeshRenderer(Node *node) :
    Renderable("Skinned Mesh Renderer", node)
{
}

void SkinnedMeshRenderer::OnImGui()
{
	//Renderable::OnImGui();
}

std::type_index SkinnedMeshRenderer::GetType() const
{
	return typeid(SkinnedMeshRenderer);
}
}        // namespace Cmpt
}        // namespace Ilum