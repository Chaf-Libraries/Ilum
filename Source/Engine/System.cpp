#include "System.hpp"

namespace Ilum
{
void System::Tick(Renderer *renderer)
{
	Execute<TransformComponent, HierarchyComponent>(renderer);
	Execute<StaticMeshComponent>(renderer);
}
}        // namespace Ilum