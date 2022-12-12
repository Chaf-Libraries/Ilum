#include "Resource.hpp"
#include "Resource/Animation.hpp"
#include "Resource/Mesh.hpp"
#include "Resource/Prefab.hpp"
#include "Resource/SkinnedMesh.hpp"
#include "Resource/Texture.hpp"

namespace Ilum
{
IResource::IResource(const std::string& name) :
    m_name(name)
{
}

const std::string &IResource::GetName() const
{
	return m_name;
}

size_t IResource::GetUUID() const
{
	return Hash(m_name);
}

template class EXPORT_API Resource<ResourceType::Mesh>;
template class EXPORT_API Resource<ResourceType::SkinnedMesh>;
template class EXPORT_API Resource<ResourceType::Texture2D>;
template class EXPORT_API Resource<ResourceType::Prefab>;
template class EXPORT_API Resource<ResourceType::Animation>;
}        // namespace Ilum