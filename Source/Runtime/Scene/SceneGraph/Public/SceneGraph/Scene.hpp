#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <typeindex>
#include <unordered_map>
#include <vector>

namespace Ilum
{
class Node;
class Component;
class RHIContext;

class Scene
{
	friend class Node;

  public:
	Scene(const std::string &name = "untitled scene");

	~Scene();

	void SetName(const std::string &name);

	const std::string &GetName() const;

	const std::vector<std::unique_ptr<Node>> &GetNodes() const;

	const std::vector<Node *> GetRoots() const;

	template <typename _Ty>
	const std::vector<_Ty *> &GetComponents() const
	{
		std::vector<_Ty *> result;

		if (HasComponent<_Ty>())
		{
			result.resize(m_components.at(typeid(_Ty)).size());

			auto &scene_components = m_components[typeid(_Ty)];

			std::transform(scene_components.begin(), scene_components.end(), result.begin(),
			               [](const std::unique_ptr<Component> &component) -> _Ty * {
				               return static_cast<_Ty *>(component.get());
			               });
		}

		return result;
	}

	template <typename _Ty>
	bool HasComponent() const
	{
		return m_components.find(typeid(_Ty)) != m_components.end();
	}

	Node *CreateNode(const std::string &name = "untitled node");

	void EraseNode(Node *node);

  private:
	std::string m_name;

	std::vector<std::unique_ptr<Node>> m_nodes;

	std::unordered_map<std::type_index, std::vector<std::unique_ptr<Component>>> m_components;
};
}        // namespace Ilum