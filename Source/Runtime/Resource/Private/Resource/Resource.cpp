#include "Resource.hpp"
#include "Resource/Animation.hpp"
#include "Resource/Mesh.hpp"
#include "Resource/Prefab.hpp"
#include "Resource/SkinnedMesh.hpp"
#include "Resource/Texture.hpp"

namespace Ilum
{
template class EXPORT_API Resource<ResourceType::Mesh>;
template class EXPORT_API Resource<ResourceType::SkinnedMesh>;
template class EXPORT_API Resource<ResourceType::Texture>;
template class EXPORT_API Resource<ResourceType::Prefab>;
template class EXPORT_API Resource<ResourceType::Animation>;
}        // namespace Ilum