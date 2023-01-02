#include "IMaterialNode.hpp"

#include <Resource/Resource/Texture2D.hpp>

using namespace Ilum;

class ImageTexture : public MaterialNode<ImageTexture>
{
	enum class Sampler
	{
		LinearClamp,
		LinearWarp,
		NearestClamp,
		NearestWarp,
	};

	std::vector<const char *> sampler_type = {
	    "LinearClamp",
	    "LinearWarp",
	    "NearestClamp",
	    "NearestWarp",
	};

	struct ImageConfig
	{
		Sampler sampler;
		char    filename[200];
	};

  public:
	virtual MaterialNodeDesc Create(size_t &handle) override
	{
		MaterialNodeDesc desc;
		return desc
		    .SetHandle(handle++)
		    .SetName("ImageTexture")
		    .SetCategory("Texture")
		    .SetVariant(ImageConfig{})
		    .Input(handle++, "UVW", MaterialNodePin::Type::Float3, MaterialNodePin::Type::RGB | MaterialNodePin::Type::Float3)
		    .Output(handle++, "Color", MaterialNodePin::Type::RGB)
		    .Output(handle++, "Alpha", MaterialNodePin::Type::Float);
	}

	virtual void OnImGui(MaterialNodeDesc &node_desc, Editor *editor) override
	{
		ImageConfig &config = *node_desc.GetVariant().Convert<ImageConfig>();

		ImGui::Combo("", (int32_t *) (&config.sampler), sampler_type.data(), static_cast<int32_t>(sampler_type.size()));

		auto *resource_manager = editor->GetRenderer()->GetResourceManager();
		if (resource_manager->Has<ResourceType::Texture2D>(config.filename))
		{
			if (ImGui::ImageButton(resource_manager->Get<ResourceType::Texture2D>(config.filename)->GetTexture(), ImVec2(100, 100)))
			{
				std::memset(config.filename, '\0', 200);
			}
		}
		else
		{
			ImGui::Button(config.filename, ImVec2(100.f, 100.f));
		}

		if (ImGui::BeginDragDropTarget())
		{
			if (const auto *pay_load = ImGui::AcceptDragDropPayload("Texture2D"))
			{
				std::memset(config.filename, '\0', 200);
				std::memcpy(config.filename, pay_load->Data, std::strlen(static_cast<const char *>(pay_load->Data)));
			}
			ImGui::EndDragDropTarget();
		}
	}

	virtual void EmitHLSL(const MaterialNodeDesc &node_desc, MaterialGraph *graph, MaterialCompilationContext &context) override
	{
	}
};

CONFIGURATION_MATERIAL_NODE(ImageTexture)