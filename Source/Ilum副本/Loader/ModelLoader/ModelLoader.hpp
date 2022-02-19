#pragma once

#include <assimp/Logger.hpp>

#include "Graphics/Model/Model.hpp"

#include "Material/PBR.h"

struct aiNode;
struct aiMesh;
struct aiScene;
struct aiMaterial;

namespace Ilum
{
class AssimpLogger : public Assimp::Logger
{
  public:
	bool attachStream(Assimp::LogStream *p_stream, uint32_t severity) override;
	bool detachStream(Assimp::LogStream *p_stream, uint32_t severity) override;

  private:
	void OnDebug(const char *message) override;

	void OnVerboseDebug(const char *message) override;

	void OnInfo(const char *message) override;

	void OnWarn(const char *message) override;

	void OnError(const char *message) override;
};

class ModelLoader
{
  public:
	ModelLoader() = default;

	~ModelLoader() = default;

	static void load(Model &model, const std::string &file_path);

  private:
	static void parseNode(const std::string &file_path, aiMatrix4x4 transform, aiNode *node, const aiScene *scene, Model &model, std::vector<uint32_t> &meshlet_offsets, std::vector<uint32_t> &meshlet_counts);
	static void parseMaterial(const std::string &file_path, aiMaterial *mesh_material, scope<material::PBRMaterial> &material);
};
}        // namespace Ilum