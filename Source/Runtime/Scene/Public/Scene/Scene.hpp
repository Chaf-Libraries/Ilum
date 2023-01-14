#pragma once

#include <Core/Core.hpp>

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

class EXPORT_API Scene
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
	std::vector<_Ty *> GetComponents()
	{
		std::vector<_Ty *> result;

		if (HasComponent<_Ty>())
		{
			auto &scene_components = GetComponents().at(typeid(_Ty));

			std::transform(scene_components.begin(), scene_components.end(), std::back_inserter(result),
			               [](const std::unique_ptr<Component> &component) -> _Ty * {
				               return static_cast<_Ty *>(component.get());
			               });
		}

		return result;
	}

	template <typename _Ty>
	bool HasComponent()
	{
		return GetComponents().find(typeid(_Ty)) != GetComponents().end();
	}

	Node *CreateNode(const std::string &name = "untitled node");

	void EraseNode(Node *node);

	void Save(OutputArchive &archive);

	void Load(InputArchive &archive);

	void Clear();

  private:
	std::unordered_map<std::type_index, std::vector<std::unique_ptr<Component>>> &GetComponents();

  private:
	struct Impl;
	Impl *m_impl = nullptr;
};
}        // namespace Ilum