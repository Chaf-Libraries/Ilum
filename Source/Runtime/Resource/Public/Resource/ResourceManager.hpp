#pragma once

#include "Resource.hpp"

#include <RHI/RHIContext.hpp>

namespace Ilum
{
class EXPORT_API ResourceManager
{
  public:
	ResourceManager(RHIContext *rhi_context);

	~ResourceManager();

	void Tick();

	template <ResourceType Type>
	Resource<Type> *Get(const std::string &name)
	{
		return static_cast<Resource<Type> *>(Get(Type, Hash(name)));
	}

	template <ResourceType Type>
	bool Has(const std::string &name)
	{
		return Has(Type, Hash(name));
	}

	template <ResourceType Type>
	size_t Index(const std::string &name)
	{
		return Index(Type, Hash(name));
	}

	template <ResourceType Type>
	void Import(const std::string &path)
	{
		Import(Type, path);
	}

	template <ResourceType Type>
	void Erase(const std::string &path)
	{
		Erase(Type, Hash(path));
	}

	template <ResourceType Type, typename... Args>
	void Add(Args &&...args)
	{
		Add(Type, std::make_unique<Resource<Type>>(std::forward<Args>(args)...));
	}

	template <ResourceType Type>
	const std::vector<std::string> GetResources() const
	{
		return GetResources(Type);
	}

	template <ResourceType Type>
	bool Update() const
	{
		return Update(Type);
	}

  private:
	IResource *Get(ResourceType type, size_t uuid);

	bool Has(ResourceType type, size_t uuid);

	size_t Index(ResourceType type, size_t uuid);

	void Import(ResourceType type, const std::string &path);

	void Erase(ResourceType type, size_t uuid);

	void Add(ResourceType type, std::unique_ptr<IResource> &&resource);

	const std::vector<std::string> GetResources(ResourceType type) const;

	bool Update(ResourceType type) const;

  private:
	struct Impl;
	Impl *m_impl = nullptr;
};
}        // namespace Ilum