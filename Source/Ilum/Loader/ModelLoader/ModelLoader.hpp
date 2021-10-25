#pragma once

#include <assimp/Logger.hpp>

#include "Graphics/Model/Model.hpp"

struct aiNode;
struct aiMesh;
struct aiScene;

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
	static void parseNode(aiMatrix4x4 &transform, aiNode *node, const aiScene *scene, std::vector<SubMesh> &meshes);
	static void parseMesh(aiMatrix4x4 &transform, aiMesh *mesh, const aiScene *scene, std::vector<Vertex> &vertices, std::vector<uint32_t> &indices);
};
}        // namespace Ilum