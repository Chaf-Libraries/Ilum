#pragma once

#include "Bone.hpp"

#include <Core/Core.hpp>

namespace Ilum
{
class EXPORT_API Animation
{
  public:
	struct Node
	{
		std::string       name;
		glm::mat4         transform;
		std::vector<Node> children;
	};

  public:
	Animation() = default;

	Animation(const std::string &name, std::vector<Bone> &&bones, float duration, float ticks_per_sec);

	Animation(Animation &&animation);

	~Animation();

	Bone *GetBone(const std::string &name);

	bool IsEmpty() const;

  private:
	struct Impl;
	Impl *m_impl = nullptr;
};
}        // namespace Ilum