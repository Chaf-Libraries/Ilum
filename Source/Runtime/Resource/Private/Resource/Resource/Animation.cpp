#include "Resource/Animation.hpp"

#include <RHI/RHIContext.hpp>

namespace Ilum
{
struct Resource<ResourceType::Animation>::Impl
{
	std::vector<Bone> bones;

	std::map<std::string, std::string> hierarchy;
};

Resource<ResourceType::Animation>::Resource(RHIContext *rhi_context, const std::string &name, std::vector<Bone> &&bones, std::map<std::string, std::string> &&hierarchy, float duration, float ticks_per_sec) :
    IResource(name)
{
	m_impl            = new Impl;
	m_impl->bones     = std::move(bones);
	m_impl->hierarchy = std::move(hierarchy);
}

const std::vector<Bone> &Resource<ResourceType::Animation>::GetBones() const
{
	return m_impl->bones;
}

Bone *Resource<ResourceType::Animation>::GetBone(const std::string &name)
{
	auto iter = std::find_if(m_impl->bones.begin(), m_impl->bones.end(), [&](const Bone &bone) { return bone.GetBoneName() == name; });
	return iter == m_impl->bones.end() ? nullptr : &(*iter);
}

uint32_t Resource<ResourceType::Animation>::GetBoneCount() const
{
	return static_cast<uint32_t>(m_impl->bones.size());
}

uint32_t Resource<ResourceType::Animation>::GetMaxBoneIndex() const
{
	uint32_t idx = 0;
	for (auto &bone : m_impl->bones)
	{
		idx = std::max(idx, bone.GetBoneID());
	}
	return idx;
}

const std::map<std::string, std::string> &Resource<ResourceType::Animation>::GetHierarchy() const
{
	return m_impl->hierarchy;
}

Resource<ResourceType::Animation>::~Resource()
{
	delete m_impl;
}

}        // namespace Ilum