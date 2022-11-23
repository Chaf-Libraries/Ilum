#include "System.hpp"

namespace Ilum
{
System &System::GetInstance()
{
	static System system;
	return system;
}

void System::Tick(Renderer *renderer)
{
	Execute<TransformComponent, HierarchyComponent>(renderer);
	Execute<StaticMeshComponent>(renderer);
}
}        // namespace Ilum