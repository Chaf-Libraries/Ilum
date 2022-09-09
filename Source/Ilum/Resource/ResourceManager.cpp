#include "ResourceManager.hpp"
#include "Importer/Texture/STBImporter.hpp"

#include <CodeGeneration/Meta/RHIMeta.hpp>
#include <Core/Path.hpp>
#include <RHI/RHIContext.hpp>

#include <filesystem>

namespace Ilum
{
inline void LoadTextureFromBuffer(RHIContext *rhi_context, std::unique_ptr<RHITexture> &texture, const TextureDesc &desc, const std::vector<uint8_t> &data, bool mipmap)
{
	texture = rhi_context->CreateTexture2D(desc.width, desc.height, desc.format, desc.usage, mipmap);

	auto staging_buffer = rhi_context->CreateBuffer(BufferDesc{"", RHIBufferUsage::Transfer, RHIMemoryUsage::GPU_TO_CPU, data.size()});
	std::memcpy(staging_buffer->Map(), data.data(), data.size());
	staging_buffer->Flush(0, data.size());
	staging_buffer->Unmap();

	auto *cmd_buffer = rhi_context->CreateCommand(RHIQueueFamily::Graphics);
	cmd_buffer->Begin();
	cmd_buffer->ResourceStateTransition({TextureStateTransition{
	                                        texture.get(),
	                                        RHIResourceState::Undefined,
	                                        RHIResourceState::TransferDest,
	                                        TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}}},
	                                    {});
	cmd_buffer->CopyBufferToTexture(staging_buffer.get(), texture.get(), 0, 0, 1);
	cmd_buffer->ResourceStateTransition({TextureStateTransition{
	                                        texture.get(),
	                                        RHIResourceState::TransferDest,
	                                        RHIResourceState::ShaderResource,
	                                        TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}}},
	                                    {});
	cmd_buffer->End();

	auto queue = rhi_context->CreateQueue(RHIQueueFamily::Graphics, 1);
	auto fence = rhi_context->CreateFence();
	queue->Submit({cmd_buffer});
	queue->Execute(fence.get());
	fence->Wait();
}

ResourceManager::ResourceManager(RHIContext *rhi_context) :
    p_rhi_context(rhi_context)
{
	// Thumbnail
	{
		std::vector<uint8_t> data;
		TextureDesc          desc;
		STBImporter::GetInstance().Import(p_rhi_context, "Asset/Icon/scene.png", data, desc);
		LoadTextureFromBuffer(p_rhi_context, m_thumbnails[ResourceType::Scene], desc, data, true);
		STBImporter::GetInstance().Import(p_rhi_context, "Asset/Icon/render_graph.png", data, desc);
		LoadTextureFromBuffer(p_rhi_context, m_thumbnails[ResourceType::RenderGraph], desc, data, true);
	}

	const std::string meta_path = "Asset/Meta";
	for (const auto &path : std::filesystem::directory_iterator(meta_path))
	{
		if (Path::GetInstance().GetFileExtension(path.path().string()) == ".meta")
		{
			std::ifstream is(path.path().string(), std::ios::binary);
			InputArchive  archive(is);
			ResourceType  type = ResourceType::Unknown;
			archive(type);
			switch (type)
			{
				case ResourceType::Unknown:
					LOG_WARN("Unknow resource type, fail to load {}", path.path().string());
					break;
				case ResourceType::Texture: {
					TextureMeta          meta = {};
					std::vector<uint8_t> thumbnail_data;
					archive(meta.uuid, meta.desc, thumbnail_data);

					LoadTextureFromBuffer(p_rhi_context, meta.thumbnail, ThumbnailDesc, thumbnail_data, false);

					m_texture_index[Path::GetInstance().GetFileName(path.path().string(), false)] = m_textures.size();
					m_textures.emplace_back(std::make_unique<TextureMeta>(std::move(meta)));
				}
				break;
				case ResourceType::Model:
					break;
				case ResourceType::Scene: {
					SceneMeta meta = {};
					archive(meta.uuid, meta.name);
					std::string uuid = meta.uuid;
					m_scenes.emplace(uuid, std::make_unique<SceneMeta>(std::move(meta)));
				}
				break;
				case ResourceType::RenderGraph: {
					RenderGraphMeta meta = {};
					archive(meta.uuid, meta.name);
					std::string uuid = meta.uuid;
					m_render_graphs.emplace(uuid, std::make_unique<RenderGraphMeta>(std::move(meta)));
				}
				break;
				default:
					break;
			}
		}
	}
}

ResourceManager::~ResourceManager()
{
}

void ResourceManager::ImportTexture(const std::string &filename)
{
	std::string uuid = std::to_string(Hash(filename));
	if (m_texture_index.find(uuid) != m_texture_index.end())
	{
		return;
	}

	std::string extension = Path::GetInstance().GetFileExtension(filename);

	TextureMeta          meta = {};
	std::vector<uint8_t> data;

	if (extension == ".png" || extension == ".jpg" || extension == ".hdr" || extension == ".bmp")
	{
		STBImporter::GetInstance().Import(p_rhi_context, filename, data, meta.desc);
	}

	if (data.empty())
	{
		return;
	}

	BufferDesc buffer_desc = {};
	buffer_desc.size       = data.size();
	buffer_desc.usage      = RHIBufferUsage::Transfer;
	buffer_desc.memory     = RHIMemoryUsage::CPU_TO_GPU;

	auto staging_buffer = p_rhi_context->CreateBuffer(buffer_desc);
	std::memcpy(staging_buffer->Map(), data.data(), buffer_desc.size);
	staging_buffer->Flush(0, buffer_desc.size);
	staging_buffer->Unmap();

	auto thumbnail_buffer = p_rhi_context->CreateBuffer(BufferDesc{"", RHIBufferUsage::Transfer, RHIMemoryUsage::GPU_TO_CPU, 48 * 48 * 4 * sizeof(uint8_t)});

	meta.texture   = p_rhi_context->CreateTexture2D(meta.desc.width, meta.desc.height, meta.desc.format, meta.desc.usage, true);
	meta.thumbnail = p_rhi_context->CreateTexture(ThumbnailDesc);
	meta.uuid      = uuid;

	{
		auto *cmd_buffer = p_rhi_context->CreateCommand(RHIQueueFamily::Graphics);
		cmd_buffer->Begin();
		cmd_buffer->ResourceStateTransition({TextureStateTransition{
		                                         meta.texture.get(),
		                                         RHIResourceState::Undefined,
		                                         RHIResourceState::TransferDest,
		                                         TextureRange{RHITextureDimension::Texture2D, 0, meta.desc.mips, 0, 1}},
		                                     TextureStateTransition{
		                                         meta.thumbnail.get(),
		                                         RHIResourceState::Undefined,
		                                         RHIResourceState::TransferDest,
		                                         TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}}},
		                                    {});
		cmd_buffer->CopyBufferToTexture(staging_buffer.get(), meta.texture.get(), 0, 0, 1);
		cmd_buffer->GenerateMipmaps(meta.texture.get(), RHIResourceState::Undefined, RHIFilter::Linear);
		cmd_buffer->ResourceStateTransition({TextureStateTransition{
		                                        meta.texture.get(),
		                                        RHIResourceState::TransferDest,
		                                        RHIResourceState::TransferSource,
		                                        TextureRange{RHITextureDimension::Texture2D, 0, meta.desc.mips, 0, 1}}},
		                                    {});
		cmd_buffer->BlitTexture(
		    meta.texture.get(),
		    TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1},
		    RHIResourceState::TransferSource,
		    meta.thumbnail.get(),
		    TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1},
		    RHIResourceState::TransferDest);
		cmd_buffer->ResourceStateTransition({TextureStateTransition{
		                                         meta.texture.get(),
		                                         RHIResourceState::TransferSource,
		                                         RHIResourceState::ShaderResource,
		                                         TextureRange{RHITextureDimension::Texture2D, 0, meta.desc.mips, 0, 1}},
		                                     TextureStateTransition{
		                                         meta.thumbnail.get(),
		                                         RHIResourceState::TransferDest,
		                                         RHIResourceState::TransferSource,
		                                         TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}}},
		                                    {});
		cmd_buffer->CopyTextureToBuffer(meta.thumbnail.get(), thumbnail_buffer.get(), 0, 0, 1);
		cmd_buffer->ResourceStateTransition({TextureStateTransition{
		                                        meta.thumbnail.get(),
		                                        RHIResourceState::TransferSource,
		                                        RHIResourceState::ShaderResource,
		                                        TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}}},
		                                    {});
		cmd_buffer->End();

		auto queue = p_rhi_context->CreateQueue(RHIQueueFamily::Graphics, 1);
		auto fence = p_rhi_context->CreateFence();
		queue->Submit({cmd_buffer});
		queue->Execute(fence.get());
		fence->Wait();
	}

	std::vector<uint8_t> thumbnail_data(thumbnail_buffer->GetDesc().size);
	std::memcpy(thumbnail_data.data(), thumbnail_buffer->Map(), thumbnail_buffer->GetDesc().size);
	thumbnail_buffer->Unmap();

	SERIALIZE("./Asset/Meta/" + uuid + ".meta", ResourceType::Texture, meta.uuid, meta.desc, thumbnail_data, data);

	m_texture_index[uuid] = m_textures.size();

	meta.index = m_texture_array.size();

	m_texture_array.push_back(meta.texture.get());
	m_textures.emplace_back(std::make_unique<TextureMeta>(std::move(meta)));
}

void ResourceManager::AddSceneMeta(const SceneMeta &meta)
{
	if (m_scenes.find(meta.uuid) == m_scenes.end())
	{
		m_scenes.emplace(meta.uuid, std::make_unique<SceneMeta>(std::move(meta)));
	}
}

void ResourceManager::AddRenderGraphMeta(const RenderGraphMeta &meta)
{
	if (m_render_graphs.find(meta.uuid) == m_render_graphs.end())
	{
		m_render_graphs.emplace(meta.uuid, std::make_unique<RenderGraphMeta>(std::move(meta)));
	}
}

const std::vector<std::unique_ptr<TextureMeta>> &ResourceManager::GetTextureMeta() const
{
	return m_textures;
}

const std::unordered_map<std::string, std::unique_ptr<SceneMeta>> &ResourceManager::GetSceneMeta() const
{
	return m_scenes;
}

const std::unordered_map<std::string, std::unique_ptr<RenderGraphMeta>> &ResourceManager::GetRenderGraphMeta() const
{
	return m_render_graphs;
}

const std::vector<RHITexture *> &ResourceManager::GetTextureArray() const
{
	return m_texture_array;
}

const TextureMeta *ResourceManager::GetTexture(const std::string &uuid)
{
	if (m_texture_index.find(uuid) != m_texture_index.end())
	{
		auto &meta = *m_textures[m_texture_index.at(uuid)];

		if (!meta.texture)
		{
			std::ifstream is("Asset/Meta/" + meta.uuid + ".meta", std::ios::binary);
			InputArchive  archive(is);

			TextureMeta          tmp_meta = {};
			std::vector<uint8_t> thumbnail_data;
			std::vector<uint8_t> texture_data;
			archive(meta.uuid, meta.desc, thumbnail_data, texture_data);

			LoadTextureFromBuffer(p_rhi_context, meta.texture, meta.desc, texture_data, true);

			meta.index = m_texture_array.size();
			m_texture_array.push_back(meta.texture.get());
		}

		return m_textures[m_texture_index.at(uuid)].get();
	}
	return nullptr;
}

const SceneMeta *ResourceManager::GetScene(const std::string &uuid)
{
	if (m_scenes.find(uuid) != m_scenes.end())
	{
		return m_scenes.at(uuid).get();
	}
	return nullptr;
}

const RenderGraphMeta *ResourceManager::GetRenderGraph(const std::string &uuid)
{
	if (m_render_graphs.find(uuid) != m_render_graphs.end())
	{
		return m_render_graphs.at(uuid).get();
	}
	return nullptr;
}

RHITexture *ResourceManager::GetThumbnail(ResourceType type) const
{
	return m_thumbnails.at(type).get();
}
}        // namespace Ilum