#include "Resource/SkinnedMesh.hpp"

namespace Ilum
{
struct Resource<ResourceType::SkinnedMesh>::Impl
{
	std::string name;
};

Resource<ResourceType::SkinnedMesh>::Resource(RHIContext *rhi_context, const std::string &name, std::vector<SkinnedVertex> &&vertices, std::vector<uint32_t> &&indices)
{
	m_impl = new Impl;
}

Resource<ResourceType::SkinnedMesh>::~Resource()
{
	delete m_impl;
}

const std::string &Resource<ResourceType::SkinnedMesh>::GetName() const
{
	return m_impl->name;
}
}        // namespace Ilum