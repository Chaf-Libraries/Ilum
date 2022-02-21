#pragma once

#include "Resource/Model/Model.hpp"
#include "Resource/Image/Bitmap.hpp"

#include <Graphics/Resource/Image.hpp>

#include <future>
#include <map>
#include <unordered_set>

namespace Ilum::Resource
{
class ResourceCache
{
  public:
	ResourceCache()  = default;
	~ResourceCache() = default;

	static void OnUpdate();

	static Graphics::ImageReference LoadTexture2D(const std::string &filepath);
	static void                     LoadTexture2DAsync(const std::string &filepath);

	static ModelReference LoadModel(const std::string &filepath);
	static void           LoadModelAsync(const std::string &filepath);

	static void RemoveTexture2D(const std::string &filepath);
	static void RemoveModel(const std::string &filepath);

	static uint32_t GetTexture2DIndex(const std::string &filepath);

	static std::vector<Graphics::ImageReference> GetTexture2DReference();
	static std::vector<ModelReference>           GetModelReference();

	static bool ImageUpdate();
	static bool ModelUpdate();

	static bool IsModelLoading();
	static bool IsTexture2DLoading();

	static void ClearTexture2D();
	static void ClearModel();
	static void ClearAll();

  private:
	static ResourceCache &Get();

  private:
	std::vector<std::unique_ptr<Graphics::Image>>                        m_images;
	std::unordered_map<std::string, uint32_t>                            m_image_query;
	std::unordered_set<std::string>                                      m_new_image_async;
	std::map<std::string, std::future<Bitmap>> m_loading_image_async;
	std::unordered_set<std::string>                                      m_deprecated_image_async;
	std::mutex                                                           m_image_async_load_mutex;
	bool                                                                 m_image_update = false;

	std::vector<std::unique_ptr<Model>>                        m_models;
	std::unordered_map<std::string, uint32_t>                  m_model_query;
	std::unordered_set<std::string>                            m_new_model_async;
	std::map<std::string, std::future<std::unique_ptr<Model>>> m_loading_model_async;
	std::unordered_set<std::string>                            m_deprecated_model_async;
	std::mutex                                                 m_model_async_load_mutex;
	bool                                                       m_model_update = false;
};
}        // namespace Ilum::Resource