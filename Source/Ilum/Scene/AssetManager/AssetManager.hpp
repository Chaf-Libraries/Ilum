#pragma once

namespace Ilum
{
class AssetManager
{
  public:
	AssetManager(RHIContext *rhi_context);

	~AssetManager();

	void ImportTexture2D(const std::string &filename);

	void ImportModel(const std::string &filename);

  private:
	const std::string m_asset_path = "./Asset/";
};
}        // namespace Ilum