#pragma once

#include "../Resource.hpp"

#include <RHI/RHIContext.hpp>

namespace Ilum
{
template <>
class EXPORT_API Resource<ResourceType::Prefab> final : public IResource
{
  public:
	template <typename Archive>
	void serialize(Archive &archive, std::pair<ResourceType, std::string> &m)
	{
		archive(m.first, m.second);
	}

	struct Node
	{
		std::string name      = "";
		glm::mat4   transform = glm::mat4(1.f);

		std::vector<Node> children;

		std::vector<std::pair<ResourceType, std::string>> resources;

		template <typename Archive>
		void save(Archive &archive) const
		{
			archive(name, transform, children);
			archive(resources.size());
			for (auto &[type, name] : resources)
			{
				archive(type, name);
			}
		}

		template <typename Archive>
		void load(Archive &archive)
		{
			archive(name, transform, children);
			size_t count = 0;
			archive(count);
			for (size_t i = 0; i < count; i++)
			{
				ResourceType type = ResourceType::Unknown;
				std::string  name;
				archive(type, name);
				resources.emplace_back(std::make_pair(type, name));
			}
		}
	};

  public:
	Resource(RHIContext *rhi_context, const std::string &name);

	Resource(RHIContext *rhi_context, const std::string &name, Node &&root);

	virtual ~Resource() override;

	virtual bool Validate() const override;

	virtual void Load(RHIContext *rhi_context) override;

	const Node &GetRoot() const;

  private:
	struct Impl;
	Impl *m_impl = nullptr;
};
}        // namespace Ilum