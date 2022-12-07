#pragma once

#include "../Resource.hpp"

#include <RHI/RHIContext.hpp>

namespace Ilum
{
template <>
class EXPORT_API Resource<ResourceType::Prefab> final : public IResource
{
  public:
	struct Node
	{
		std::string name;
		glm::mat4   transform;

		std::vector<Node> children;

		std::vector<std::pair<ResourceType, std::string>> resources;
	};

  public:
	Resource(const std::string &name, Node &&root);

	virtual ~Resource() override;

	const Node &GetRoot() const;

  private:
	struct Impl;
	Impl *m_impl = nullptr;
};
}        // namespace Ilum