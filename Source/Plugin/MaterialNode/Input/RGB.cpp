#include "IMaterialNode.hpp"

using namespace Ilum;

class RGB : public MaterialNode<RGB>
{
  public:
	virtual MaterialNodeDesc Create(size_t &handle) override
	{
		MaterialNodeDesc desc;
		return desc
		    .SetHandle(handle++)
		    .SetName("RGB")
		    .SetCategory("Input")
		    .Output(handle++, "Color", MaterialNodePin::Type::RGB, glm::vec3(0.f));
	}

	virtual void OnImGui(MaterialNodeDesc &node_desc, Editor *editor) override
	{
	}

	virtual void EmitHLSL(const MaterialNodeDesc &node_desc, const MaterialGraphDesc &graph_desc, Renderer *renderer, MaterialCompilationContext *context) override
	{
		if (context->IsCompiled(node_desc))
		{
			return;
		}

		glm::vec3 color = *node_desc.GetPin("Color").variant.Convert<glm::vec3>();
		context->variables.emplace_back(fmt::format("float3 S_{} = float3({}, {}, {});", node_desc.GetPin("Color").handle, color.x, color.y, color.z));
	}
};

CONFIGURATION_MATERIAL_NODE(RGB)