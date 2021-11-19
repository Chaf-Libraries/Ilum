#include "ResourceCache.hpp"

#include "Loader/ImageLoader/ImageLoader.hpp"
#include "Loader/ModelLoader/ModelLoader.hpp"

#include "Device/LogicalDevice.hpp"
#include "Graphics/GraphicsContext.hpp"
#include "Graphics/Shader/ShaderCompiler.hpp"

#include "File/FileSystem.hpp"

#include "Threading/ThreadPool.hpp"

#include "Graphics/Vulkan/VK_Debugger.h"

namespace Ilum
{
ImageReference ResourceCache::loadImage(const std::string &filepath)
{
	if (m_image_cache.size() == m_image_map.size() && m_image_map.find(filepath) != m_image_map.end())
	{
		return m_image_cache.at(m_image_map.at(filepath));
	}

	LOG_INFO("Import Image: {}", filepath);

	Image image;
	ImageLoader::loadImageFromFile(image, filepath);

	{
		std::lock_guard<std::mutex> lock(m_image_mutex);
		m_new_image[filepath] = std::move(image);
		//m_image_cache.emplace_back(std::move(image));
		//m_image_map[filepath] = m_image_cache.size() - 1;
	}

	return m_image_cache.back();
}

void ResourceCache::loadImageAsync(const std::string &filepath)
{
	ThreadPool::instance()->addTask([this, filepath](size_t) {
		if (m_image_map.find(filepath) == m_image_map.end())
		{
			LOG_INFO("Import Image: {} using thread {}", filepath, ThreadPool::instance()->threadIndex());

			Image image;
			ImageLoader::loadImageFromFile(image, filepath);
			VK_Debugger::setName(image.getView(), filepath.c_str());

			{
				std::lock_guard<std::mutex> lock(m_image_mutex);
				m_new_image[filepath] = std::move(image);
				/*m_image_cache.emplace_back(std::move(image));
				m_image_map[filepath] = m_image_cache.size() - 1;*/
			}
		}
	});
}

void ResourceCache::removeImage(const std::string &filepath)
{
	if (!hasImage(filepath))
	{
		return;
	}

	m_deprecated_image.push_back(filepath);
}

bool ResourceCache::hasImage(const std::string &filepath) const
{
	return m_image_map.find(filepath) != m_image_map.end();
}

const std::unordered_map<std::string, size_t> &ResourceCache::getImages()
{
	return m_image_map;
}

const std::vector<ImageReference> ResourceCache::getImageReferences()
{
	std::vector<ImageReference> references;
	references.reserve(m_image_map.size());

	for (auto &image : m_image_cache)
	{
		references.push_back(image);
	}

	return references;
}

uint32_t ResourceCache::imageID(const std::string &filepath)
{
	if (!hasImage(filepath))
	{
		return std::numeric_limits<uint32_t>::max();
	}

	return static_cast<uint32_t>(m_image_map.at(filepath));
}

ModelReference ResourceCache::loadModel(const std::string &name)
{
	if (m_model_cache.size() == m_model_map.size() && m_model_map.find(name) != m_model_map.end())
	{
		return m_model_cache.at(m_model_map.at(name));
	}

	Model model;
	ModelLoader::load(model, name);

	std::lock_guard<std::mutex> lock(m_model_mutex);
	m_new_model[name] = std::move(model);

	//m_model_cache.emplace_back(std::move(model));
	//m_model_map[name] = m_model_cache.size() - 1;

	LOG_INFO("Import Model: {}", name);

	return m_model_cache.back();
}

void ResourceCache::loadModelAsync(const std::string &filepath)
{
	ThreadPool::instance()->addTask([this, filepath](size_t) {
		std::string name = filepath;

		std::lock_guard<std::mutex> lock(m_model_mutex);
		while (m_model_map.find(name) != m_model_map.end() || m_new_model.find(name) != m_new_model.end())
		{
			name += "#";
		}

		LOG_INFO("Import Image: {} using thread #{}", filepath, ThreadPool::instance()->threadIndex());

		Model model;
		ModelLoader::load(model, filepath);

		m_new_model[name] = std::move(model);
		//m_model_cache.emplace_back(std::move(model));
		//m_model_map[name] = m_model_cache.size() - 1;
	});
}

void ResourceCache::removeModel(const std::string &filepath)
{
	if (!hasModel(filepath))
	{
		return;
	}

	m_deprecated_model.push_back(filepath);
}

bool ResourceCache::hasModel(const std::string &filepath)
{
	return m_model_map.find(filepath) != m_model_map.end();
}

const std::unordered_map<std::string, size_t> &ResourceCache::getModels()
{
	return m_model_map;
}

void ResourceCache::flush()
{
	if (m_deprecated_model.empty() && m_deprecated_image.empty() && m_new_image.empty() && m_new_model.empty())
	{
		return;
	}
	else
	{
		GraphicsContext::instance()->getQueueSystem().waitAll();
	}

	{
		std::lock_guard<std::mutex> lock(m_model_mutex);

		// Remove deprecated model
		for (auto &name : m_deprecated_model)
		{
			size_t index = m_model_map.at(name);
			std::swap(m_model_cache.begin() + index, m_model_cache.begin() + m_model_cache.size() - 1);
			for (auto &[name, idx] : m_model_map)
			{
				if (idx == m_model_cache.size() - 1)
				{
					idx = index;
				}
			}
			m_model_cache.erase(m_model_cache.begin() + m_model_cache.size() - 1);
			m_model_map.erase(name);
			LOG_INFO("Release Model: {}", name);
		}
		m_deprecated_model.clear();

		// Add new model
		for (auto &[name, model] : m_new_model)
		{
			m_model_map[name] = m_model_cache.size();
			m_model_cache.push_back(std::move(model));
		}
		m_new_model.clear();
	}

	{
		std::lock_guard<std::mutex> lock(m_image_mutex);

		// Remove deprecated image
		for (auto &name : m_deprecated_image)
		{
			size_t index = m_image_map.at(name);
			std::swap(m_image_cache.begin() + index, m_image_cache.begin() + m_image_cache.size() - 1);
			for (auto &[name, idx] : m_image_map)
			{
				if (idx == m_image_cache.size() - 1)
				{
					idx = index;
				}
			}
			m_image_cache.erase(m_image_cache.begin() + index);
			m_image_map.erase(name);
			LOG_INFO("Release Image: {}", name);
		}
		m_deprecated_image.clear();

		// Add new model
		for (auto &[name, image] : m_new_image)
		{
			m_image_map[name] = m_image_cache.size();
			m_image_cache.push_back(std::move(image));
		}
		m_new_image.clear();
	}
}
}        // namespace Ilum