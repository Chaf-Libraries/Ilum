#include "IMaterialNode.hpp"

using namespace Ilum;

class MaskedMaterial : public MaterialNode<MaskedMaterial>
{
  public:
	virtual MaterialNodeDesc Create(size_t &handle) override
	{
		MaterialNodeDesc desc;
		return desc
		    .SetHandle(handle++)
		    .SetName("MaskedMaterial")
		    .SetCategory("BSDF")
		    .Input(handle++, "BSDF", MaterialNodePin::Type::BSDF, MaterialNodePin::Type::BSDF)
		    .Input(handle++, "Opacity", MaterialNodePin::Type::Float, MaterialNodePin::Type::Float | MaterialNodePin::Type::Float3 | MaterialNodePin::Type::RGB, float(0.f))
		    .Input(handle++, "Threshold", MaterialNodePin::Type::Float, MaterialNodePin::Type::Float | MaterialNodePin::Type::Float3 | MaterialNodePin::Type::RGB, float(0.f))
		    .Output(handle++, "Out", MaterialNodePin::Type::BSDF);
	}

	virtual void OnImGui(MaterialNodeDesc &node_desc, Editor *editor) override
	{
	}

	virtual void EmitHLSL(const MaterialNodeDesc &node_desc, const MaterialGraphDesc &graph_desc, ResourceManager *manager, MaterialCompilationContext *context) override
	{
		if (context->IsCompiled(node_desc))
		{
			return;
		}

		std::map<std::string, std::string> parameters;

		context->SetParameter<float>(parameters, node_desc.GetPin("Opacity"), graph_desc, manager, context);
		context->SetParameter<float>(parameters, node_desc.GetPin("Threshold"), graph_desc, manager, context);

		auto &bsdf_node = node_desc.GetPin("BSDF");

		if (graph_desc.HasLink(bsdf_node.handle))
		{
			size_t bsdf = 0;

			{
				auto &src_node = graph_desc.GetNode(graph_desc.LinkFrom(bsdf_node.handle));
				src_node.EmitHLSL(graph_desc, manager, context);
				auto &src_pin = src_node.GetPin(graph_desc.LinkFrom(bsdf_node.handle));

				bsdf = src_pin.handle;
			}

			{
				std::string bsdf_type = "";

				std::string bsdf_name = fmt::format("S_{}", bsdf);

				for (auto &bsdf : context->bsdfs)
				{
					if (bsdf.name == bsdf_name)
					{
						bsdf_type = bsdf.type;
						break;
					}
				}

				context->bsdfs.emplace_back(MaterialCompilationContext::BSDF{
				    fmt::format("S_{}", node_desc.GetPin("Out").handle),
				    fmt::format("MaskedMaterial< {} >", bsdf_type),
				    fmt::format("S_{}.Init(S_{}, {}, {});", node_desc.GetPin("Out").handle,
				                bsdf,
				                parameters["Opacity"],
				                parameters["Threshold"])});
			}
		}
	}
};

CONFIGURATION_MATERIAL_NODE(MaskedMaterial)