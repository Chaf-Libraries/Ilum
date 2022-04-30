#include "ResourceCache.hpp"

#include "Loader/ImageLoader/ImageLoader.hpp"
#include "Loader/ModelLoader/ModelLoader.hpp"

#include "Device/LogicalDevice.hpp"

#include "Graphics/GraphicsContext.hpp"
#include "Graphics/Shader/ShaderCompiler.hpp"

#include "File/FileSystem.hpp"

#include "Threading/ThreadPool.hpp"

#include "Graphics/Vulkan/VK_Debugger.h"

#include "Renderer/Renderer.hpp"

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
		m_image_cache.emplace_back(std::move(image));
		m_image_map[filepath] = m_image_cache.size() - 1;
	}

	return m_image_cache.back();
}

void ResourceCache::loadImageAsync(const std::string &filepath)
{
	std::lock_guard<std::mutex> lock(m_image_mutex);
	if (FileSystem::isExist(filepath))
	{
		m_new_image.insert(filepath);
	}
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

void ResourceCache::updateImageReferences()
{
	m_image_references.clear();
	m_image_references.reserve(m_image_map.size());

	for (auto &image : m_image_cache)
	{
		m_image_references.push_back(image);
	}
}

const std::vector<ImageReference> &ResourceCache::getImageReferences() const
{
	return m_image_references;
}

uint32_t ResourceCache::imageID(const std::string &filepath) const
{
	if (!hasImage(filepath))
	{
		return std::numeric_limits<uint32_t>::max();
	}

	return static_cast<uint32_t>(m_image_map.at(filepath));
}

void ResourceCache::clearImages()
{
	for (auto &[name, idx] : m_image_map)
	{
		m_deprecated_image.push_back(name);
	}
}

bool ResourceCache::isLoadingImage() const
{
	return !m_image_futures.empty();
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
	m_model_cache.emplace_back(std::move(model));
	m_model_map[name] = m_model_cache.size() - 1;

	m_vertices_count += m_model_cache.back().vertices_count;
	m_indices_count += m_model_cache.back().indices_count;
	//m_meshlet_vertices_count += m_model_cache.back().meshlet_vertices_count;
	//m_meshlet_indices_count += m_model_cache.back().meshlet_indices_count;

	LOG_INFO("Import Model: {}", name);

	return m_model_cache.back();
}

void ResourceCache::loadModelAsync(const std::string &filepath)
{
	std::lock_guard<std::mutex> lock(m_model_mutex);
	if (FileSystem::isExist(filepath))
	{
		m_new_model.insert(filepath);
	}
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

bool ResourceCache::isLoadingModel() const
{
	return !m_model_futures.empty();
}

void ResourceCache::clearModels()
{
	for (auto &[name, idx] : m_model_map)
	{
		m_deprecated_model.push_back(name);
	}
}

const uint32_t ResourceCache::getVerticesCount() const
{
	return m_vertices_count;
}

const uint32_t ResourceCache::getMeshletVerticesCount() const
{
	return m_meshlet_vertices_count;
}

const uint32_t ResourceCache::getIndicesCount() const
{
	return m_indices_count;
}

const uint32_t ResourceCache::getMeshletIndicesCount() const
{
	return m_meshlet_indices_count;
}

void ResourceCache::clear()
{
	clearImages();
	clearModels();
}

void ResourceCache::flush()
{
	if (m_deprecated_model.empty() && m_deprecated_image.empty() &&
	    m_new_image.empty() && m_new_model.empty() &&
	    m_image_futures.empty() && m_model_futures.empty())
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
			std::iter_swap(m_model_cache.begin() + index, m_model_cache.begin() + m_model_cache.size() - 1);
			for (auto &[name, idx] : m_model_map)
			{
				if (idx == m_model_cache.size() - 1)
				{
					idx = index;
				}
			}

			auto &model = m_model_cache.begin() + m_model_cache.size() - 1;
			m_vertices_count -= model->vertices_count;
			m_indices_count -= model->indices_count;
			//m_meshlet_vertices_count -= model->meshlet_vertices_count;
			//m_meshlet_indices_count -= model->meshlet_indices_count;

			m_model_cache.erase(model);
			m_model_map.erase(name);
			LOG_INFO("Release Model: {}", name);
		}
		m_deprecated_model.clear();

		// Add new model
		for (auto &filepath : m_new_model)
		{
			if (m_model_futures.find(filepath) == m_model_futures.end() && m_model_map.find(filepath) == m_model_map.end())
			{
				m_model_futures[filepath] = ThreadPool::instance()->addTask([filepath](size_t id) {
					Model model;
					LOG_INFO("Loading model {}, using thread {}", filepath, id);
					ModelLoader::load(model, filepath);
					return model;
				});
			}
		}
		m_new_model.clear();

		// Get sync models
		for (auto iter = m_model_futures.begin(); iter != m_model_futures.end();)
		{
			auto &[name, future] = *iter;
			if (future.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
			{
				m_model_map[name] = m_model_cache.size();
				m_model_cache.push_back(std::move(future.get()));
				m_vertices_count += m_model_cache.back().vertices_count;
				m_indices_count += m_model_cache.back().indices_count;
				//m_meshlet_vertices_count += m_model_cache.back().meshlet_vertices_count;
				//m_meshlet_indices_count += m_model_cache.back().meshlet_indices_count;
				iter = m_model_futures.erase(iter);
			}
			else
			{
				iter++;
			}
		}

		// Update model vertex offset
		{
			uint32_t vertices_offset = 0;
			uint32_t indices_offset  = 0;
			uint32_t meshlet_vertices_offset = 0;
			uint32_t meshlet_indices_offset = 0;
			for (auto &[name, idx] : m_model_map)
			{
				auto &model = m_model_cache[idx];

				model.vertices_offset = vertices_offset;
				model.indices_offset  = indices_offset;

				//model.meshlet_vertices_offset = meshlet_vertices_offset;
				//model.meshlet_indices_offset  = meshlet_indices_offset;

				vertices_offset += model.vertices_count;
				indices_offset += model.indices_count;

				//meshlet_vertices_offset += model.meshlet_vertices_count;
				//meshlet_indices_offset += model.meshlet_indices_count;
			}
		}
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
			Material::update = true;
		}
		m_deprecated_image.clear();

		// Add new image
		for (auto &filepath : m_new_image)
		{
			if (m_image_futures.find(filepath) == m_image_futures.end() && m_image_map.find(filepath) == m_image_map.end())
			{
				m_image_futures[filepath] = ThreadPool::instance()->addTask([filepath](size_t id) {
					Image image;
					ImageLoader::loadImageFromFile(image, filepath);
					LOG_INFO("Loading image {}, using thread {}", filepath, id);
					return image;
				});
			}
		}

		for (auto iter = m_image_futures.begin(); iter != m_image_futures.end();)
		{
			auto &[name, future] = *iter;
			if (future.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
			{
				m_image_map[name] = m_image_cache.size();
				m_image_cache.push_back(std::move(future.get()));
				iter = m_image_futures.erase(iter);
				Material::update = true;
			}
			else
			{
				iter++;
			}
		}

		m_new_image.clear();
	}
}
}        // namespace Ilum