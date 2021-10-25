#include "AssetBrowser.hpp"

#include "Renderer/Renderer.hpp"

#include "ImGui/ImGuiContext.hpp"

#include "Loader/ImageLoader/ImageLoader.hpp"

#include "Graphics/GraphicsContext.hpp"
#include "Graphics/Pipeline/ShaderCache.hpp"

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
		if (ImGui::IsItemHovered() && ImGui::IsWindowFocused())
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
		ImGui::ImageButton(
		    ImGuiContext::textureID(image, Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp)),
		    {height / static_cast<float>(image.getHeight()) * static_cast<float>(image.getWidth()), height});

		// Drag&Drop source
		if (ImGui::BeginDragDropSource())
		{
			ImGui::SetDragDropPayload("Model", &name, sizeof(std::string));
			ImGui::EndDragDropSource();
		}

		// Image Hint
		if (ImGui::IsItemHovered() && ImGui::IsWindowFocused())
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
		draw_texture_asset(100.f, 0.f);
	}
	else if (current_item == 1)
	{
		draw_model_asset(m_model_icon, 100.f, 0.f);
	}
	else if (current_item == 2)
	{
		draw_shader_asset(m_shader_icon, 100.f, 0.f);
	}

	ImGui::End();
}
}        // namespace Ilum::panel