#pragma once

#include <Core/Core.hpp>

#include <memory>
#include <string>
#include <typeindex>
#include <unordered_map>

namespace Ilum
{
class Component;
class Scene;

class EXPORT_API Node
{
  public:
	Node(size_t id, Scene &scene, const std::string &name = "untitled node");

	~Node();

	size_t GetID() const;

	void SetName(const std::string &name);

	const std::string &GetName() const;

	Node *GetParent();

	void SetParent(Node *node);

	const std::vector<Node *> &GetChildren() const;

	template <typename _Ty>
	bool HasComponent()
	{
		return HasComponent_(typeid(_Ty));
	}

	template <typename _Ty>
	void EraseComponent()
	{
		return EraseComponent(typeid(_Ty));
	}

	void EraseComponent(std::type_index index);

	template <typename _Ty, typename = std::is_base_of<Component, _Ty>>
	_Ty *GetComponent()
	{
		return static_cast<_Ty *>(GetComponent_(typeid(_Ty)));
	}

	template <typename _Ty, typename = std::is_base_of<Component, _Ty>>
	_Ty *AddComponent(std::unique_ptr<_Ty> &&component)
	{
		return static_cast<_Ty *>(AddComponent_(std::move(component)));
	}

	const std::unordered_map<std::type_index, Component *> &GetComponents() const;

  private:
	void EraseChild(Node *node);

	void AddChild(Node *node);

	bool HasComponent_(std::type_index index);

	Component *GetComponent_(std::type_index index);

	Component *AddComponent_(std::unique_ptr<Component> &&component);

	void AddComponent_(Component *component);

  private:
	struct Impl;
	Impl *m_impl = nullptr;
};
}        // namespace Ilum