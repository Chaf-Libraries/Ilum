#pragma once

#include "Resource.hpp"

#include <RHI/RHIContext.hpp>

namespace Ilum
{
class ResourceManager
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
	RHITexture *GetThumbnail(const std::string &name)
	{
		return GetThumbnail(Type, Hash(name));
	}

	template <ResourceType Type>
	bool Valid(const std::string &name)
	{
		return Valid(Type, Hash(name));
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
	const std::vector<std::string> GetResources(bool only_valid = true) const
	{
		return GetResources(Type, only_valid);
	}

	template <ResourceType Type>
	bool Update() const
	{
		return Update(Type);
	}

	template <ResourceType Type>
	void SetDirty()
	{
		SetDirty(Type);
	}

  private:
	IResource *Get(ResourceType type, size_t uuid);

	RHITexture* GetThumbnail(ResourceType type, size_t uuid);

	bool Valid(ResourceType type, size_t uuid);

	bool Has(ResourceType type, size_t uuid);

	size_t Index(ResourceType type, size_t uuid);

	void Import(ResourceType type, const std::string &path);

	void Erase(ResourceType type, size_t uuid);

	void Add(ResourceType type, std::unique_ptr<IResource> &&resource);

	const std::vector<std::string> GetResources(ResourceType type, bool only_valid) const;

	bool Update(ResourceType type) const;

	void SetDirty(ResourceType type);

  private:
	struct Impl;
	Impl *m_impl = nullptr;
};
}        // namespace Ilum