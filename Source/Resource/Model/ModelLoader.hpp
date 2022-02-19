#pragma once

#include "Model.hpp"

#include "Material/Material.hpp"

#include <assimp/matrix4x4.h>

#include <glm/glm.hpp>

#include <memory>

struct aiNode;
struct aiMesh;
struct aiScene;
struct aiMaterial;

namespace Ilum::Resource
{
class ModelLoader
{
  public:
	ModelLoader()  = default;
	~ModelLoader() = default;

	static std::unique_ptr<Model> Load(const std::string &file_path);

  private:
	static void ParseNode(const std::string &file_path, aiMatrix4x4 &transform, const aiNode *node, const aiScene *scene, Model &model, std::vector<uint32_t> &meshlet_offsets, std::vector<uint32_t> &meshlet_counts);
	static void ParseMaterial(const std::string &file_path, const aiMaterial *mesh_material, Material &material);
};
}        // namespace Ilum::Resource