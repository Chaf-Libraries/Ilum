#include "Animation.hpp"

namespace Ilum
{
struct Animation::Impl
{
	std::string name;
	float       duration;
	float       ticks_per_sec;

	std::vector<Bone> bones;
};

Animation::Animation(const std::string &name, std::vector<Bone> &&bones, float duration, float ticks_per_sec)
{
	m_impl = new Impl;

	m_impl->name          = name;
	m_impl->duration      = duration;
	m_impl->ticks_per_sec = ticks_per_sec;
	m_impl->bones         = std::move(bones);

}

Animation::Animation(Animation &&animation) noexcept :
    m_impl(animation.m_impl)
{
	animation.m_impl = nullptr;
}

Animation::~Animation()
{
	if (m_impl)
	{
		delete m_impl;
	}
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