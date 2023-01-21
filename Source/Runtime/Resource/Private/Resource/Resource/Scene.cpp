#include "Resource/Scene.hpp"

#include <Scene/Scene.hpp>

namespace Ilum
{
Resource<ResourceType::Scene>::Resource(RHIContext *rhi_context, const std::string &name) :
    IResource(rhi_context, name, ResourceType::Scene)
{
}

Resource<ResourceType::Scene>::Resource(RHIContext *rhi_context, const std::string &name, Scene *scene) :
    IResource(name)
{
	Save(scene);
}

void Resource<ResourceType::Scene>::Update(Scene *scene)
{
	std::vector<uint32_t> thumbnail_data;

	scene->Clear();

	{
		std::ifstream is(fmt::format("Asset/Meta/{}.{}.asset", m_name, (uint32_t) ResourceType::Scene), std::ios::binary);
		InputArchive input_archive(is);
		input_archive(thumbnail_data);
		scene->Load(input_archive);
	}
}

void Resource<ResourceType::Scene>::Save(Scene *scene)
{
	std::vector<uint32_t> thumbnail_data;

	{
		std::ofstream os(fmt::format("Asset/Meta/{}.{}.asset", m_name, (uint32_t) ResourceType::Scene), std::ios::binary);
		OutputArchive output_archive(os);
		output_archive(thumbnail_data);
		scene->Save(output_archive);
	}
}
}        // namespace Ilum