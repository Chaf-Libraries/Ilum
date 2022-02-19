#pragma once

#include "ResourceCache.hpp"
#include "Image/ImageLoader.hpp"
#include "Model/ModelLoader.hpp"

#include <Graphics/Device/Device.hpp>
#include <Graphics/RenderContext.hpp>

#include <Core/FileSystem.hpp>
#include <Core/JobSystem/JobSystem.hpp>

namespace Ilum::Resource
{
void ResourceCache::OnUpdate()
{
	// Handle texture2D
	// Remove deprecated image
	if (!Get().m_deprecated_image_async.empty())
	{
		Graphics::RenderContext::WaitDevice();
		for (auto &name : Get().m_deprecated_image_async)
		{
			if (Get().m_image_query.find(name) != Get().m_image_query.end())
			{
				uint32_t index = Get().m_image_query.at(name);
				std::swap(Get().m_images.begin() + index, Get().m_images.begin() + Get().m_images.size() - 1);
				for (auto &[name, idx] : Get().m_image_query)
				{
					if (idx == Get().m_images.size() - 1)
					{
						idx = index;
					}
				}
				Get().m_images.erase(Get().m_images.begin() + index);
				Get().m_image_query.erase(name);
				LOG_INFO("Release Image: {}", name);
			}
		}
		Get().m_deprecated_image_async.clear();
	}

	// Async loading new image on worker threads
	if (!Get().m_new_image_async.empty())
	{
		for (auto &filepath : Get().m_new_image_async)
		{
			if (Get().m_image_query.find(filepath) == Get().m_image_query.end())
			{
				Core::JobHandle handle;
				Get().m_loading_image_async.emplace(filepath, Core::JobSystem::Execute(
				                                                  handle, [path = filepath]() {
					                                                  auto &cmd_buffer = Graphics::RenderContext::CreateCommandBuffer();
					                                                  auto  image      = ImageLoader::LoadTexture2DFromFile(Graphics::RenderContext::GetDevice(), cmd_buffer, path);
					                                                  Graphics::RenderContext::ResetCommandPool();
					                                                  return image;
				                                                  }));
			}
		}
		Get().m_new_image_async.clear();
	}

	// Collect async loading images
	for (auto iter = Get().m_loading_image_async.begin(); iter != Get().m_loading_image_async.end();)
	{
		if (iter->second.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
		{
			Get().m_images.emplace_back(std::move(iter->second.get()));
			Get().m_image_query.emplace(iter->first, static_cast<uint32_t>(Get().m_images.size()) - 1);
			iter = Get().m_loading_image_async.erase(iter);
		}
		else
		{
			iter++;
		}
	}

	// Handle model
	// Remove deprecated model
	if (!Get().m_deprecated_model_async.empty())
	{
		Graphics::RenderContext::WaitDevice();
		for (auto &name : Get().m_deprecated_model_async)
		{
			if (Get().m_model_query.find(name) != Get().m_model_query.end())
			{
				uint32_t index = Get().m_model_query.at(name);
				std::swap(Get().m_models.begin() + index, Get().m_models.begin() + Get().m_models.size() - 1);
				for (auto &[name, idx] : Get().m_model_query)
				{
					if (idx == Get().m_models.size() - 1)
					{
						idx = index;
					}
				}
				Get().m_models.erase(Get().m_models.begin() + index);
				Get().m_model_query.erase(name);
				LOG_INFO("Release Image: {}", name);
			}
		}
		Get().m_deprecated_model_async.clear();
	}

	// Async loading new model on worker threads
	if (!Get().m_new_model_async.empty())
	{
		for (auto &filepath : Get().m_new_model_async)
		{
			if (Get().m_model_query.find(filepath) == Get().m_model_query.end())
			{
				Core::JobHandle handle;
				Get().m_loading_model_async.emplace(filepath, Core::JobSystem::Execute(handle, [&filepath]() {
					                                    return ModelLoader::Load(filepath);
				                                    }));
			}
		}
		Get().m_new_model_async.clear();
	}

	// Collect async loading models
	for (auto iter = Get().m_loading_model_async.begin(); iter != Get().m_loading_model_async.end();)
	{
		if (iter->second.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
		{
			Get().m_models.emplace_back(std::move(iter->second.get()));
			Get().m_model_query.emplace(iter->first, static_cast<uint32_t>(Get().m_models.size()) - 1);
			iter = Get().m_loading_model_async.erase(iter);
		}
		else
		{
			iter++;
		}
	}
}

Graphics::ImageReference ResourceCache::LoadTexture2D(const std::string &filepath)
{
	if (Get().m_image_query.find(filepath) != Get().m_image_query.end())
	{
		return *Get().m_images.at(Get().m_image_query.at(filepath));
	}

	auto &cmd_buffer = Graphics::RenderContext::CreateCommandBuffer();
	Get().m_images.emplace_back(ImageLoader::LoadTexture2DFromFile(Graphics::RenderContext::GetDevice(), cmd_buffer, filepath));
	Graphics::RenderContext::ResetCommandPool();

	Get().m_image_query.emplace(filepath, static_cast<uint32_t>(Get().m_images.size()) - 1);

	Graphics::RenderContext::SetName(*Get().m_images.at(Get().m_image_query.at(filepath)), filepath.c_str());

	return *Get().m_images.at(Get().m_image_query.at(filepath));
}

void ResourceCache::LoadTexture2DAsync(const std::string &filepath)
{
	if (Get().m_loading_image_async.find(filepath) == Get().m_loading_image_async.end() ||
	    Get().m_deprecated_image_async.find(filepath) == Get().m_deprecated_image_async.end())
	{
		Get().m_new_image_async.insert(filepath);
	}
}

ModelReference ResourceCache::LoadModel(const std::string &filepath)
{
	if (Get().m_model_query.find(filepath) != Get().m_model_query.end())
	{
		return *Get().m_models.at(Get().m_model_query.at(filepath));
	}

	Get().m_models.emplace_back(ModelLoader::Load(filepath));
	Get().m_model_query.emplace(filepath, static_cast<uint32_t>(Get().m_models.size()) - 1);

	return *Get().m_models.at(Get().m_model_query.at(filepath));
}

void ResourceCache::LoadModelAsync(const std::string &filepath)
{
	if (Get().m_loading_model_async.find(filepath) == Get().m_loading_model_async.end() ||
	    Get().m_deprecated_model_async.find(filepath) == Get().m_deprecated_model_async.end())
	{
		Get().m_new_model_async.insert(filepath);
	}
}

void ResourceCache::RemoveTexture2D(const std::string &filepath)
{
	Get().m_deprecated_image_async.insert(filepath);
}

void ResourceCache::RemoveModel(const std::string &filepath)
{
	Get().m_deprecated_model_async.insert(filepath);
}

uint32_t ResourceCache::GetTexture2DIndex(const std::string &filepath)
{
	if (Get().m_image_query.find(filepath) != Get().m_image_query.end())
	{
		return Get().m_image_query.at(filepath);
	}
	return std::numeric_limits<uint32_t>::max();
}

std::vector<Graphics::ImageReference> ResourceCache::GetTexture2DReference()
{
	std::vector<Graphics::ImageReference> image_reference;
	for (auto &image : Get().m_images)
	{
		image_reference.push_back(*image);
	}
	return image_reference;
}

std::vector<ModelReference> ResourceCache::GetModelReference()
{
	std::vector<ModelReference> model_reference;
	for (auto &model : Get().m_models)
	{
		model_reference.push_back(*model);
	}
	return model_reference;
}

bool ResourceCache::HasNewModelLoaded()
{
	return !Get().m_loading_image_async.empty();
}

bool ResourceCache::HasNewTexture2DLoaded()
{
	return !Get().m_loading_model_async.empty();
}

bool ResourceCache::IsModelLoading()
{
	return !Get().m_loading_model_async.empty();
}

bool ResourceCache::IsTexture2DLoading()
{
	return !Get().m_loading_image_async.empty();
}

void ResourceCache::ClearTexture2D()
{
	std::lock_guard<std::mutex> lock(Get().m_image_async_load_mutex);

	Get().m_images.clear();
	Get().m_image_query.clear();
	Get().m_new_image_async.clear();
	Get().m_loading_image_async.clear();
	Get().m_deprecated_image_async.clear();
}

void ResourceCache::ClearModel()
{
	std::lock_guard<std::mutex> lock(Get().m_model_async_load_mutex);

	Get().m_models.clear();
	Get().m_model_query.clear();
	Get().m_new_model_async.clear();
	Get().m_loading_model_async.clear();
	Get().m_deprecated_model_async.clear();
}

void ResourceCache::ClearAll()
{
	ClearTexture2D();
	ClearModel();
}

ResourceCache &ResourceCache::Get()
{
	static ResourceCache resource_cache;
	return resource_cache;
}
}        // namespace Ilum::Resource