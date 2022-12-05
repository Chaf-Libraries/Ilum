#include "Resource/Animation.hpp"

#include <Animation/Animation.hpp>

namespace Ilum
{
struct Resource<ResourceType::Animation>::Impl
{
	Animation animation;
};

Resource<ResourceType::Animation>::Resource(Animation &&animation)
{
	m_impl = new Impl;
}

Resource<ResourceType::Animation>::~Resource()
{
	delete m_impl;
}

const Animation &Resource<ResourceType::Animation>::GetAnimation() const
{
	return m_impl->animation;
}
}        // namespace Ilum