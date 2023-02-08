#include "IMaterialNode.hpp"
#include "Utils/SPDLoader.hpp"

using namespace Ilum;

class ConductorBSDF : public MaterialNode<ConductorBSDF>
{
	const std::vector<const char *> m_materials = {
	    "a-C", "Ag", "Al", "AlAs", "AlSb", "a-SiH", "Au", "Be", "Cr", "Csl", "Cu",
	    "Cu2O", "CuO", "d-C", "Hg", "HgTe", "Ir", "K", "KBr", "KCl", "Li", "MgO", "Mo",
	    "NaCl", "Nb", "Rh", "Se", "Se-e", "SiC", "SiO", "SnTe", "Ta", "Te", "Te-e", "ThF4",
	    "TiC", "TiN", "TiO2", "TiO2-e", "VC", "VN", "W"};

  public:
	virtual MaterialNodeDesc Create(size_t &handle) override
	{
		MaterialNodeDesc desc;
		return desc
		    .SetHandle(handle++)
		    .SetName("ConductorBSDF")
		    .SetCategory("BSDF")
		    .SetVariant(int32_t(1))
		    .Input(handle++, "Reflectance", MaterialNodePin::Type::RGB, MaterialNodePin::Type::Float | MaterialNodePin::Type::RGB | MaterialNodePin::Type::Float3, glm::vec3(1.f))
		    .Input(handle++, "Roughness", MaterialNodePin::Type::Float, MaterialNodePin::Type::Float | MaterialNodePin::Type::RGB | MaterialNodePin::Type::Float3, float(0.1f))
		    .Input(handle++, "Normal", MaterialNodePin::Type::Float3, MaterialNodePin::Type::RGB | MaterialNodePin::Type::Float3)
		    .Output(handle++, "Out", MaterialNodePin::Type::BSDF);
	}

	virtual void OnImGui(MaterialNodeDesc &node_desc, Editor *editor) override
	{
		int32_t &material_type = *node_desc.GetVariant().Convert<int32_t>();

	ImGui::PushID("material type");
		if (ImGui::BeginCombo("", "Type"))
		{
			for (size_t i = 0; i < m_materials.size(); i++)
			{
				const bool is_selected = (i == material_type);
				if (ImGui::Selectable(m_materials[i], i == material_type))
				{
					material_type = static_cast<int32_t>(i);
				}
				if (is_selected)
				{
					ImGui::SetItemDefaultFocus();
				}
			}
			ImGui::EndCombo();
		}
		ImGui::PopID();
	}

	virtual void EmitHLSL(const MaterialNodeDesc &node_desc, const MaterialGraphDesc &graph_desc, ResourceManager *manager, MaterialCompilationContext *context) override
	{
		if (context->IsCompiled(node_desc))
		{
			return;
		}

		int32_t material_type = *node_desc.GetVariant().Convert<int32_t>();

		std::map<std::string, std::string> parameters;

		if (!context->HasParameter<glm::vec3>(parameters, node_desc.GetPin("Normal"), graph_desc, manager, context))
		{
			parameters["Normal"] = "surface_interaction.isect.n";
		}
		else
		{
			parameters["Normal"] = fmt::format("ExtractNormalMap(surface_interaction, {})", parameters["Normal"]);
		}

		context->SetParameter<glm::vec3>(parameters, node_desc.GetPin("Reflectance"), graph_desc, manager, context);
		context->SetParameter<float>(parameters, node_desc.GetPin("Roughness"), graph_desc, manager, context);

		glm::vec3 eta = SPDLoader::Load(fmt::format("Asset/SPD/metals/{}.eta.spd", m_materials[material_type]));
		glm::vec3 k   = SPDLoader::Load(fmt::format("Asset/SPD/metals/{}.k.spd", m_materials[material_type]));

		context->bsdfs.emplace_back(MaterialCompilationContext::BSDF{
		    fmt::format("S_{}", node_desc.GetPin("Out").handle),
		    "ConductorBSDF",
		    fmt::format("S_{}.Init({}, {}, float3({}, {}, {}),  float3({}, {}, {}), {});", node_desc.GetPin("Out").handle,
		                parameters["Reflectance"],
		                parameters["Roughness"],
		                eta.x, eta.y, eta.z,
		                k.x, k.y, k.z,
		                parameters["Normal"])});
	}
};

CONFIGURATION_MATERIAL_NODE(ConductorBSDF)