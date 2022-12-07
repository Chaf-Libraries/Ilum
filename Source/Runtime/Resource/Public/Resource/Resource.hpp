#pragma once

#include <Core/Core.hpp>

namespace Ilum
{
enum class ResourceType
{
	Prefab,
	Mesh,
	SkinnedMesh,
	Texture,
	Animation,
	Material,
};

class EXPORT_API IResource
{
  public:
	explicit IResource(const std::string &name);

	virtual ~IResource() = default;

	const std::string &GetName() const;

	size_t GetUUID() const;

  private:
	const std::string m_name;
};

template <ResourceType Type>
class EXPORT_API Resource : public IResource
{
  public:
	Resource() = default;

	virtual ~Resource() override = default;
};
}        // namespace Ilum