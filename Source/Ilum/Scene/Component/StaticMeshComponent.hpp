#pragma once

#include "Component.hpp"

#include <Resource/Importer/Model/ModelImporter.hpp>

#include <vector>

namespace Ilum
{
struct StaticMeshComponent : public Component
{
	std::string uuid;

	std::vector<Submesh> submeshes;

	[[serialization(false), reflection(false)]] std::shared_ptr<RHIBuffer> per_instance_buffer = nullptr;
};
}        // namespace Ilum