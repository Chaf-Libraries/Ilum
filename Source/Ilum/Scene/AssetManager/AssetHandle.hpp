#pragma once

namespace Ilum
{
enum class AssetType
{
	Mesh,
	Material,
	Texture2D,

	Scene,
	RenderGraph
};

class AssetHandle
{
  public:
	AssetHandle(AssetType type);

	~AssetHandle() = default;

  private:
	AssetType type;
};
}        // namespace Ilum