#include "Animation.hpp"

namespace Ilum
{
struct Animation::Impl
{
	std::string       name;
	Node              root;
	std::vector<Bone> bones;
};

Animation::Animation(const std::string &name, std::vector<Bone> &&bones, Node &&root)
{
	m_impl = new Impl;

	m_impl->name  = name;
	m_impl->root  = std::move(root);
	m_impl->bones = std::move(bones);
}

Animation::~Animation()
{
	if (m_impl)
	{
		delete m_impl;
	}
}

const Animation::Node &Animation::GetRoot() const
{
	return m_impl->root;
}

Bone *Animation::GetBone(const std::string &name)
{
	auto iter = std::find_if(m_impl->bones.begin(), m_impl->bones.end(), [&](const Bone &bone) { return bone.GetBoneName() == name; });
	return iter == m_impl->bones.end() ? nullptr : &(*iter);
}

bool Animation::IsEmpty() const
{
	return m_impl == nullptr;
}
}        // namespace Ilum