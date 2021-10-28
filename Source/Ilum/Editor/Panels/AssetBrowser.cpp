#include "AssetBrowser.hpp"

#include "Renderer/Renderer.hpp"

#include "ImGui/ImGuiContext.hpp"

#include "Loader/ImageLoader/ImageLoader.hpp"

#include "Graphics/GraphicsContext.hpp"
#include "Graphics/Pipeline/ShaderCache.hpp"

#include "Scene/Component/MeshRenderer.hpp"
#include "Scene/Entity.hpp"
#include "Scene/Scene.hpp"

#include "ImFileDialog.h"

#include <imgui.h>

namespace Ilum::panel
{
inline void draw_texture_asset(float height, float space)
{
	auto &image_cache = Renderer::instance()->getResourceCache().getImages();

	float width = 0.f;

	ImGuiStyle &style       = ImGui::GetStyle();
	style.ItemSpacing       = ImVec2(10.f, 10.f);
	float window_visible_x2 = ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMax().x;

	for (auto &[name, index] : image_cache)
	{
		auto image = Renderer::instance()->getResourceCache().loadImage(name);

		ImGui::ImageButton(
		    ImGuiContext::textureID(image, Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp)),
		    {height / static_cast<float>(image.get().getHeight()) * static_cast<float>(image.get().getWidth()), height});

		// Drag&Drop source
		if (ImGui::BeginDragDropSource())
		{
			if (image.get().getLayerCount() == 1)
			{
				ImGui::SetDragDropPayload("Texture2D", &name, sizeof(std::string));
			}
			else if (image.get().getLayerCount() == 6)
			{
				ImGui::SetDragDropPayload("TextureCube", &name, sizeof(std::string));
				ImGui::SetDragDropPayload("TextureArray", &name, sizeof(std::string));
			}
			else
			{
				ImGui::SetDragDropPayload("TextureArray", &name, sizeof(std::string));
			}
			ImGui::EndDragDropSource();
		}

		// Image Hint
		if (ImGui::BeginPopupContextItem(name.c_str()))
		{
			if (ImGui::MenuItem("Delete"))
			{
				Renderer::instance()->getResourceCache().removeImage(name);
			}
			ImGui::EndPopup();
		}
		else if (ImGui::IsItemHovered() && ImGui::IsWindowFocused())
		{
			ImVec2 pos = ImGui::GetIO().MousePos;
			ImGui::SetNextWindowPos(ImVec2(pos.x + 10.f, pos.y + 10.f));
			ImGui::Begin(name.c_str(), NULL, ImGuiWindowFlags_Tooltip | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar);
			ImGui::Text(name.c_str());
			ImGui::Separator();
			ImGui::Text("format: %s", std::to_string(image.get().getFormat()).c_str());
			ImGui::Text("width: %s", std::to_string(image.get().getWidth()).c_str());
			ImGui::Text("height: %s", std::to_string(image.get().getHeight()).c_str());
			ImGui::Text("mip levels: %s", std::to_string(image.get().getMipLevelCount()).c_str());
			ImGui::Text("layers: %s", std::to_string(image.get().getLayerCount()).c_str());
			ImGui::End();
		}

		float last_button_x2 = ImGui::GetItemRectMax().x;
		float next_button_x2 = last_button_x2 + style.ItemSpacing.x + height / static_cast<float>(image.get().getHeight()) * static_cast<float>(image.get().getWidth());
		if (next_button_x2 < window_visible_x2)
		{
			ImGui::SameLine();
		}
	}
}

inline void draw_model_asset(const Image &image, float height, float space)
{
	auto &model_cache = Renderer::instance()->getResourceCache().getModels();

	ImGuiStyle &style       = ImGui::GetStyle();
	style.ItemSpacing       = ImVec2(10.f, 10.f);
	float window_visible_x2 = ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMax().x;

	for (auto &[name, index] : model_cache)
	{
		ImGui::PushID(name.c_str());
		ImGui::ImageButton(
		    ImGuiContext::textureID(image, Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp)),
		    {height / static_cast<float>(image.getHeight()) * static_cast<float>(image.getWidth()), height});

		// Drag&Drop source
		if (ImGui::BeginDragDropSource())
		{
			ImGui::SetDragDropPayload("Model", &name, sizeof(std::string));
			ImGui::EndDragDropSource();
		}

		ImGui::PopID();

		// Image Hint
		if (ImGui::BeginPopupContextItem(name.c_str()))
		{
			if (ImGui::MenuItem("Delete"))
			{
				auto view = Scene::instance()->getRegistry().view<cmpt::MeshRenderer>();
				for (auto &entity : view)
				{
					auto &mesh_renderer = view.get<cmpt::MeshRenderer>(entity);
					if (mesh_renderer.model == name)
					{
						mesh_renderer.model = "";
					}
				}
				Renderer::instance()->getResourceCache().removeModel(name);
			}
			ImGui::EndPopup();
		}
		else if (ImGui::IsItemHovered() && ImGui::IsWindowFocused())
		{
			auto   model = Renderer::instance()->getResourceCache().loadModel(name);
			ImVec2 pos   = ImGui::GetIO().MousePos;
			ImGui::SetNextWindowPos(ImVec2(pos.x + 10.f, pos.y + 10.f));
			ImGui::Begin(name.c_str(), NULL, ImGuiWindowFlags_Tooltip | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar);
			ImGui::Text(name.c_str());
			ImGui::Separator();
			for (uint32_t i = 0; i < model.get().getSubMeshes().size(); i++)
			{
				ImGui::Text("SubMesh #%d", i);
				ImGui::BulletText("vertices count: %d", model.get().getSubMeshes()[i].getVertexCount());
				ImGui::BulletText("indices count: %d", model.get().getSubMeshes()[i].getIndexCount());
				ImGui::BulletText("index offset: %d", model.get().getSubMeshes()[i].getIndexOffset());
			}
			ImGui::End();
		}

		float last_button_x2 = ImGui::GetItemRectMax().x;
		float next_button_x2 = last_button_x2 + style.ItemSpacing.x + height / static_cast<float>(image.getHeight()) * static_cast<float>(image.getWidth());
		if (next_button_x2 < window_visible_x2)
		{
			ImGui::SameLine();
		}
	}
}

inline void draw_shader_asset(const Image &image, float height, float space)
{
	auto &shader_cache = GraphicsContext::instance()->getShaderCache().getShaders();

	ImGuiStyle &style       = ImGui::GetStyle();
	style.ItemSpacing       = ImVec2(10.f, 10.f);
	float window_visible_x2 = ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMax().x;

	for (auto &[name, index] : shader_cache)
	{
		ImGui::ImageButton(
		    ImGuiContext::textureID(image, Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp)),
		    {height / static_cast<float>(image.getHeight()) * static_cast<float>(image.getWidth()), height});

		// Drag&Drop source
		if (ImGui::BeginDragDropSource())
		{
			ImGui::SetDragDropPayload("Shader", &name, sizeof(std::string));
			ImGui::EndDragDropSource();
		}

		// Image Hint
		if (ImGui::IsItemHovered() && ImGui::IsWindowFocused())
		{
			auto   shader_data = GraphicsContext::instance()->getShaderCache().reflect(GraphicsContext::instance()->getShaderCache().getShader(name));
			ImVec2 pos         = ImGui::GetIO().MousePos;
			ImGui::SetNextWindowPos(ImVec2(pos.x + 10.f, pos.y + 10.f));
			ImGui::Begin(name.c_str(), NULL, ImGuiWindowFlags_Tooltip | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar);
			ImGui::Text(name.c_str());
			ImGui::Separator();
			if (!shader_data.attributes.empty())
			{
				ImGui::Text("Attribute");
				if (ImGui::BeginTable("shader attribute", 7, ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders))
				{
					ImGui::TableSetupColumn("name");
					ImGui::TableSetupColumn("location");
					ImGui::TableSetupColumn("vec_size");
					ImGui::TableSetupColumn("array_size");
					ImGui::TableSetupColumn("columns");
					ImGui::TableSetupColumn("type");
					ImGui::TableSetupColumn("stage");
					ImGui::TableHeadersRow();

					for (auto &attribute : shader_data.attributes)
					{
						ImGui::TableNextRow();
						ImGui::TableSetColumnIndex(0);
						ImGui::Text("%s", attribute.name.c_str());
						ImGui::TableSetColumnIndex(1);
						ImGui::Text("%s", std::to_string(attribute.location).c_str());
						ImGui::TableSetColumnIndex(2);
						ImGui::Text("%s", std::to_string(attribute.vec_size).c_str());
						ImGui::TableSetColumnIndex(3);
						ImGui::Text("%s", std::to_string(attribute.array_size).c_str());
						ImGui::TableSetColumnIndex(4);
						ImGui::Text("%s", std::to_string(attribute.columns).c_str());
						ImGui::TableSetColumnIndex(5);
						switch (attribute.type)
						{
							case ReflectionData::Attribute::Type::Input:
								ImGui::Text("Input");
								break;
							case ReflectionData::Attribute::Type::Output:
								ImGui::Text("Output");
								break;
							case ReflectionData::Attribute::Type::None:
								ImGui::Text("None");
								break;
							default:
								break;
						}
						ImGui::TableSetColumnIndex(6);
						ImGui::Text("%s", std::to_string(attribute.stage).c_str());
					}
					ImGui::EndTable();
				}
			}

			if (!shader_data.input_attachments.empty())
			{
				ImGui::Text("Input Attachment");
				if (ImGui::BeginTable("input attachment", 7, ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders))
				{
					ImGui::TableSetupColumn("name");
					ImGui::TableSetupColumn("set");
					ImGui::TableSetupColumn("binding");
					ImGui::TableSetupColumn("index");
					ImGui::TableSetupColumn("array_size");
					ImGui::TableSetupColumn("bindless");
					ImGui::TableSetupColumn("stage");
					ImGui::TableHeadersRow();

					for (auto &input_attachment : shader_data.input_attachments)
					{
						ImGui::TableNextRow();
						ImGui::TableSetColumnIndex(0);
						ImGui::Text("%s", input_attachment.name.c_str());
						ImGui::TableSetColumnIndex(1);
						ImGui::Text("%s", std::to_string(input_attachment.set).c_str());
						ImGui::TableSetColumnIndex(2);
						ImGui::Text("%s", std::to_string(input_attachment.binding).c_str());
						ImGui::TableSetColumnIndex(3);
						ImGui::Text("%s", std::to_string(input_attachment.input_attachment_index).c_str());
						ImGui::TableSetColumnIndex(4);
						ImGui::Text("%s", std::to_string(input_attachment.array_size).c_str());
						ImGui::TableSetColumnIndex(5);
						ImGui::Text("%s", input_attachment.bindless ? "true" : "false");
						ImGui::TableSetColumnIndex(6);
						ImGui::Text("%s", std::to_string(input_attachment.stage).c_str());
					}
					ImGui::EndTable();
				}
			}

			if (!shader_data.constants.empty())
			{
				ImGui::Text("Constants");
				if (ImGui::BeginTable("shader constants", 6, ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders))
				{
					ImGui::TableSetupColumn("name");
					ImGui::TableSetupColumn("size");
					ImGui::TableSetupColumn("offset");
					ImGui::TableSetupColumn("id");
					ImGui::TableSetupColumn("type");
					ImGui::TableSetupColumn("stage");
					ImGui::TableHeadersRow();

					for (auto &constant : shader_data.constants)
					{
						ImGui::TableNextRow();
						ImGui::TableSetColumnIndex(0);
						ImGui::Text("%s", constant.name.c_str());
						ImGui::TableSetColumnIndex(1);
						ImGui::Text("%s", std::to_string(constant.size).c_str());
						ImGui::TableSetColumnIndex(2);
						ImGui::Text("%s", std::to_string(constant.offset).c_str());

						ImGui::TableSetColumnIndex(3);
						if (constant.type == ReflectionData::Constant::Type::Specialization)
						{
							ImGui::Text("%s", std::to_string(constant.constant_id).c_str());
						}

						ImGui::TableSetColumnIndex(4);
						switch (constant.type)
						{
							case ReflectionData::Constant::Type::None:
								ImGui::Text("None");
								break;
							case ReflectionData::Constant::Type::Push:
								ImGui::Text("Push");
								break;
							case ReflectionData::Constant::Type::Specialization:
								ImGui::Text("Specialization");
								break;
							default:
								break;
						}
						ImGui::TableSetColumnIndex(5);
						ImGui::Text("%s", std::to_string(constant.stage).c_str());
					}
					ImGui::EndTable();
				}
			}

			if (!shader_data.buffers.empty())
			{
				ImGui::Text("Buffer");
				if (ImGui::BeginTable("shader buffer", 8, ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders))
				{
					ImGui::TableSetupColumn("name");
					ImGui::TableSetupColumn("set");
					ImGui::TableSetupColumn("binding");
					ImGui::TableSetupColumn("size");
					ImGui::TableSetupColumn("array_size");
					ImGui::TableSetupColumn("bindless");
					ImGui::TableSetupColumn("type");
					ImGui::TableSetupColumn("stage");
					ImGui::TableHeadersRow();

					for (auto &buffer : shader_data.buffers)
					{
						ImGui::TableNextRow();
						ImGui::TableSetColumnIndex(0);
						ImGui::Text("%s", buffer.name.c_str());
						ImGui::TableSetColumnIndex(1);
						ImGui::Text("%s", std::to_string(buffer.set).c_str());
						ImGui::TableSetColumnIndex(2);
						ImGui::Text("%s", std::to_string(buffer.binding).c_str());
						ImGui::TableSetColumnIndex(3);
						ImGui::Text("%s", std::to_string(buffer.size).c_str());
						ImGui::TableSetColumnIndex(4);
						ImGui::Text("%s", std::to_string(buffer.array_size).c_str());
						ImGui::TableSetColumnIndex(5);
						ImGui::Text("%s", buffer.bindless ? "true" : "false");
						ImGui::TableSetColumnIndex(6);
						switch (buffer.type)
						{
							case ReflectionData::Buffer::Type::None:
								ImGui::Text("None");
								break;
							case ReflectionData::Buffer::Type::Uniform:
								ImGui::Text("Uniform");
								break;
							case ReflectionData::Buffer::Type::Storage:
								ImGui::Text("Storage");
								break;
							default:
								break;
						}
						ImGui::TableSetColumnIndex(7);
						ImGui::Text("%s", shader_stage_to_string(buffer.stage).c_str());
					}
					ImGui::EndTable();
				}
			}

			if (!shader_data.images.empty())
			{
				ImGui::Text("Image");
				if (ImGui::BeginTable("shader image", 7, ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders))
				{
					ImGui::TableSetupColumn("name");
					ImGui::TableSetupColumn("set");
					ImGui::TableSetupColumn("binding");
					ImGui::TableSetupColumn("array_size");
					ImGui::TableSetupColumn("bindless");
					ImGui::TableSetupColumn("type");
					ImGui::TableSetupColumn("stage");
					ImGui::TableHeadersRow();

					for (auto &image : shader_data.images)
					{
						ImGui::TableNextRow();
						ImGui::TableSetColumnIndex(0);
						ImGui::Text("%s", image.name.c_str());
						ImGui::TableSetColumnIndex(1);
						ImGui::Text("%s", std::to_string(image.set).c_str());
						ImGui::TableSetColumnIndex(2);
						ImGui::Text("%s", std::to_string(image.binding).c_str());
						ImGui::TableSetColumnIndex(3);
						ImGui::Text("%s", std::to_string(image.array_size).c_str());
						ImGui::TableSetColumnIndex(4);
						ImGui::Text("%s", image.bindless ? "true" : "false");
						ImGui::TableSetColumnIndex(5);
						switch (image.type)
						{
							case ReflectionData::Image::Type::None:
								ImGui::Text("None");
								break;
							case ReflectionData::Image::Type::Image:
								ImGui::Text("Image");
								break;
							case ReflectionData::Image::Type::Sampler:
								ImGui::Text("Sampler");
								break;
							case ReflectionData::Image::Type::ImageSampler:
								ImGui::Text("ImageSampler");
								break;
							case ReflectionData::Image::Type::ImageStorage:
								ImGui::Text("ImageStorage");
								break;
							default:
								break;
						}
						ImGui::TableSetColumnIndex(6);
						ImGui::Text("%s", shader_stage_to_string(image.stage).c_str());
					}
					ImGui::EndTable();
				}
			}

			ImGui::End();
		}

		float last_button_x2 = ImGui::GetItemRectMax().x;
		float next_button_x2 = last_button_x2 + style.ItemSpacing.x + height / static_cast<float>(image.getHeight()) * static_cast<float>(image.getWidth());
		if (next_button_x2 < window_visible_x2)
		{
			ImGui::SameLine();
		}
	}
}

AssetBrowser::AssetBrowser()
{
	m_name = "Asset Browser";
	ImageLoader::loadImageFromFile(m_model_icon, std::string(PROJECT_SOURCE_DIR) + "Asset/Texture/Icon/model.png");
	ImageLoader::loadImageFromFile(m_shader_icon, std::string(PROJECT_SOURCE_DIR) + "Asset/Texture/Icon/shader.png");
}

void AssetBrowser::draw()
{
	ImGui::Begin("Asset Browser", &active);

	auto region_width = ImGui::GetContentRegionAvailWidth();

	static const char *ASSET_TYPE[] = {"Texture", "Model", "Shader"};
	static int         current_item = 0;

	ImGui::Combo("Assets", &current_item, ASSET_TYPE, 3);

	if (current_item == 0)
	{
		ImGui::SameLine();
		if (ImGui::Button("Import"))
		{
			ifd::FileDialog::Instance().Open("TextureOpenDialog", "Import Texture", "Image file (*.png;*.jpg;*.jpeg;*.bmp;*.tga;*.hdr){.png,.jpg,.jpeg,.bmp,.tga,.hdr},.*");
		}

		if (ifd::FileDialog::Instance().IsDone("TextureOpenDialog"))
		{
			if (ifd::FileDialog::Instance().HasResult())
			{
				std::string path = ifd::FileDialog::Instance().GetResult().u8string();
				Renderer::instance()->getResourceCache().loadImageAsync(path);
			}
			ifd::FileDialog::Instance().Close();
		}

		draw_texture_asset(100.f, 0.f);
	}
	else if (current_item == 1)
	{
		ImGui::SameLine();
		if (ImGui::Button("Import"))
		{
			ifd::FileDialog::Instance().Open("ModelOpenDialog", "Import Model", "Model file (*.obj;*.fbx;*.gltf){.obj,.fbx,.gltf},.*");
		}

		if (ifd::FileDialog::Instance().IsDone("ModelOpenDialog"))
		{
			if (ifd::FileDialog::Instance().HasResult())
			{
				std::string path = ifd::FileDialog::Instance().GetResult().u8string();
				Renderer::instance()->getResourceCache().loadModelAsync(path);
			}
			ifd::FileDialog::Instance().Close();
		}

		draw_model_asset(m_model_icon, 100.f, 0.f);
	}
	else if (current_item == 2)
	{
		draw_shader_asset(m_shader_icon, 100.f, 0.f);
	}

	ImGui::End();
}
}        // namespace Ilum::panel