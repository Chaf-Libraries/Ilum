#include "IMaterialNode.hpp"

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
		Sampler     sampler  = Sampler::LinearClamp;
		std::string filename = "";
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

	virtual void OnImGui(MaterialNodeDesc &node_desc) override
	{
		ImageConfig &config = *node_desc.GetVariant().Convert<ImageConfig>();
		ImGui::Combo("", (int32_t *) (&config.sampler), sampler_type.data(), static_cast<int32_t>(sampler_type.size()));
		if (ImGui::Button(config.filename.c_str(), ImVec2(100.f, 20.f)))
		{
			config.filename = "";
		}
		if (ImGui::BeginDragDropTarget())
		{
			if (const auto *pay_load = ImGui::AcceptDragDropPayload("Texture2D"))
			{
				config.filename = static_cast<const char *>(pay_load->Data);
			}
			ImGui::EndDragDropTarget();
		}
	}

	virtual void EmitHLSL(const MaterialNodeDesc &node_desc, MaterialGraph *graph) override
	{
	}
};

CONFIGURATION_MATERIAL_NODE(ImageTexture)