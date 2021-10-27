#include "ResourceCache.hpp"

#include "Loader/ImageLoader/ImageLoader.hpp"
#include "Loader/ModelLoader/ModelLoader.hpp"

#include "Device/LogicalDevice.hpp"
#include "Graphics/GraphicsContext.hpp"
#include "Graphics/Pipeline/ShaderCompiler.hpp"

#include "File/FileSystem.hpp"

#include "Threading/ThreadPool.hpp"

namespace Ilum
{
ImageReference ResourceCache::loadImage(const std::string &filepath)
{
	if (m_image_map.find(filepath) != m_image_map.end())
	{
		return m_image_cache.at(m_image_map.at(filepath));
	}

	m_image_map[filepath] = m_image_cache.size();
	m_image_cache.emplace_back(Image());
	ImageLoader::loadImageFromFile(m_image_cache.back(), filepath);

	return m_image_cache.back();
}

void ResourceCache::loadImageAsync(const std::string &filepath)
{
	ThreadPool::instance()->addTask([this, filepath](size_t) {
		if (m_image_map.find(filepath) == m_image_map.end())
		{
			Image image;
			ImageLoader::loadImageFromFile(image, filepath);

			{
				std::lock_guard<std::mutex> lock(m_image_mutex);
				m_image_map[filepath] = m_image_cache.size();
				m_image_cache.emplace_back(std::move(image));
			}
		}
	});
}

bool ResourceCache::hasImage(const std::string &filepath)
{
	return m_image_map.find(filepath) != m_image_map.end();
}

const std::unordered_map<std::string, size_t> &ResourceCache::getImages() const
{
	return m_image_map;
}

ModelReference ResourceCache::loadModel(const std::string &filepath)
{
	if (m_model_map.find(filepath) != m_model_map.end())
	{
		return m_model_cache.at(m_model_map.at(filepath));
	}

	m_model_map[filepath] = m_model_cache.size();
	m_model_cache.emplace_back(Model());
	ModelLoader::load(m_model_cache.back(), filepath);

	return m_model_cache.back();
}

void ResourceCache::loadModelAsync(const std::string &filepath)
{
	ThreadPool::instance()->addTask([this, filepath](size_t) {
		if (m_model_map.find(filepath) == m_model_map.end())
		{
			Model model;
			ModelLoader::load(model, filepath);

			{
				std::lock_guard<std::mutex> lock(m_model_mutex);
				m_model_map[filepath] = m_model_cache.size();
				m_model_cache.emplace_back(std::move(model));
			}
		}
	});
}

bool ResourceCache::hasModel(const std::string &filepath)
{
	return m_model_map.find(filepath) != m_model_map.end();
}

const std::unordered_map<std::string, size_t> &ResourceCache::getModels() const
{
	return m_model_map;
}
}        // namespace Ilum