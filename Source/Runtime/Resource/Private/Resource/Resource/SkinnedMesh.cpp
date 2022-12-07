#include "Resource/SkinnedMesh.hpp"

namespace Ilum
{
struct Resource<ResourceType::SkinnedMesh>::Impl
{
};

Resource<ResourceType::SkinnedMesh>::Resource(RHIContext *rhi_context, const std::string &name, std::vector<SkinnedVertex> &&vertices, std::vector<uint32_t> &&indices):
    IResource(name)
{
	m_impl = new Impl;
}

Resource<ResourceType::SkinnedMesh>::~Resource()
{
	delete m_impl;
}
}        // namespace Ilum