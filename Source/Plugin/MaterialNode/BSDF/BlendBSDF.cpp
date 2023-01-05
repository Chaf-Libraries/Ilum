#include "IMaterialNode.hpp"

using namespace Ilum;

class BlendBSDF : public MaterialNode<BlendBSDF>
{
  public:
	virtual MaterialNodeDesc Create(size_t &handle) override
	{
		MaterialNodeDesc desc;
		return desc
		    .SetHandle(handle++)
		    .SetName("BlendBSDF")
		    .SetCategory("BSDF")
		    .Input(handle++, "X", MaterialNodePin::Type::BSDF, MaterialNodePin::Type::BSDF)
		    .Input(handle++, "Y", MaterialNodePin::Type::BSDF, MaterialNodePin::Type::BSDF)
		    .Input(handle++, "Weight", MaterialNodePin::Type::Float, MaterialNodePin::Type::Float | MaterialNodePin::Type::Float3 | MaterialNodePin::Type::RGB, float(0.f))
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

		context->SetParameter<float>(parameters, node_desc.GetPin("Weight"), graph_desc, manager, context);

		auto &bsdf1_node = node_desc.GetPin("X");
		auto &bsdf2_node = node_desc.GetPin("Y");

		if (graph_desc.HasLink(bsdf1_node.handle) && graph_desc.HasLink(bsdf2_node.handle))
		{
			size_t bsdf1 = 0;
			size_t bsdf2 = 0;

			{
				auto &src_node = graph_desc.GetNode(graph_desc.LinkFrom(bsdf1_node.handle));
				src_node.EmitHLSL(graph_desc, manager, context);
				auto &src_pin = src_node.GetPin(graph_desc.LinkFrom(bsdf1_node.handle));

				bsdf1 = src_pin.handle;
			}

			{
				auto &src_node = graph_desc.GetNode(graph_desc.LinkFrom(bsdf2_node.handle));
				src_node.EmitHLSL(graph_desc, manager, context);
				auto &src_pin = src_node.GetPin(graph_desc.LinkFrom(bsdf2_node.handle));

				bsdf2 = src_pin.handle;
			}

			{
				std::string bsdf1_type = "";
				std::string bsdf2_type = "";

				std::string bsdf1_name = fmt::format("S_{}", bsdf1);
				std::string bsdf2_name = fmt::format("S_{}", bsdf2);

				for (auto& bsdf : context->bsdfs)
				{
					if (bsdf.name == bsdf1_name)
					{
						bsdf1_type = bsdf.type;
					}
					if (bsdf.name == bsdf2_name)
					{
						bsdf2_type = bsdf.type;
					}
					if (!bsdf1_type.empty() && !bsdf2_type.empty())
					{
						break;
					}
				}

				context->bsdfs.emplace_back(MaterialCompilationContext::BSDF{
				    fmt::format("S_{}", node_desc.GetPin("Out").handle),
				    fmt::format("BlendBSDF< {}, {} >", bsdf1_type, bsdf2_type),
				    fmt::format("S_{}.Init(S_{}, S_{}, {});", node_desc.GetPin("Out").handle, bsdf1, bsdf2, parameters["Weight"])});
			}
		}
	}
};

CONFIGURATION_MATERIAL_NODE(BlendBSDF)