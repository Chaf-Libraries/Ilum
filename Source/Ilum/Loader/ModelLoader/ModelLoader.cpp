#include "ModelLoader.hpp"

#include "File/FileSystem.hpp"

#include "Material/DisneyPBR.h"

#include <assimp/DefaultLogger.hpp>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <glm/gtc/quaternion.hpp>

namespace Ilum
{
inline glm::mat4 to_matrix(const aiMatrix4x4 &matrix)
{
	return glm::mat4(
	    matrix.a1, matrix.a2, matrix.a3, matrix.a4,
	    matrix.b1, matrix.b2, matrix.b3, matrix.b4,
	    matrix.c1, matrix.c2, matrix.c3, matrix.c4,
	    matrix.d1, matrix.d2, matrix.d3, matrix.d4);
}

inline glm::mat3 to_matrix(const aiMatrix3x3 &matrix)
{
	return glm::mat3(
	    matrix.a1, matrix.a2, matrix.a3,
	    matrix.b1, matrix.b2, matrix.b3,
	    matrix.c1, matrix.c2, matrix.c3);
}

inline glm::vec2 to_vector(const aiVector2D &vec)
{
	return glm::vec2(vec.x, vec.y);
}

inline glm::vec3 to_vector(const aiVector3D &vec)
{
	return glm::vec3(vec.x, vec.y, vec.z);
}

inline glm::vec3 to_vector(const aiColor3D &color)
{
	return glm::vec3(color.r, color.g, color.b);
}

inline glm::vec4 to_vector(const aiColor4D &color)
{
	return glm::vec4(color.r, color.g, color.b, color.a);
}

inline glm::quat to_quaternion(const aiQuaternion &quat)
{
	return glm::quat(quat.x, quat.y, quat.z, quat.w);
}

inline uint32_t get_node_count(const aiNode *node)
{
	if (!node)
	{
		return 0;
	}

	uint32_t count = 1;

	for (uint32_t i = 0; i < node->mNumChildren; i++)
	{
		count += get_node_count(node->mChildren[i]);
	}

	return count;
}

bool AssimpLogger::attachStream(Assimp::LogStream *p_stream, uint32_t severity)
{
	return true;
}

bool AssimpLogger::detachStream(Assimp::LogStream *p_stream, uint32_t severity)
{
	return true;
}

void AssimpLogger::OnDebug(const char *message)
{
#ifdef _DEBUG
	LOG_INFO(message);
#endif        // _DEBUG
}

void AssimpLogger::OnVerboseDebug(const char *message)
{
#ifdef _DEBUG
	LOG_INFO(message);
#endif        // _DEBUG
}

void AssimpLogger::OnInfo(const char *message)
{
	LOG_INFO(message);
}

void AssimpLogger::OnWarn(const char *message)
{
	LOG_WARN(message);
}

void AssimpLogger::OnError(const char *message)
{
	LOG_ERROR(message);
}

void ModelLoader::load(Model &model, const std::string &file_path)
{
	if (!FileSystem::isFile(file_path))
	{
		LOG_ERROR("Model {} is not existed!", file_path);
		return;
	}

	Assimp::Importer importer;
	Assimp::DefaultLogger::set(new AssimpLogger());

	std::vector<SubMesh>     meshes;
	std::vector<std::string> materials;

	const auto importer_flag =
	    aiProcess_FlipUVs |
	    aiProcess_FlipWindingOrder |
	    aiProcess_CalcTangentSpace |
	    aiProcess_GenSmoothNormals |
	    aiProcess_JoinIdenticalVertices |
	    aiProcess_ImproveCacheLocality |
	    aiProcess_Triangulate |
	    aiProcess_GenUVCoords |
	    aiProcess_SortByPType |
	    aiProcess_FindInvalidData |
	    aiProcess_FindInstances |
	    aiProcess_ValidateDataStructure;

	// TODO: Animation
	if (const aiScene *scene = importer.ReadFile(file_path, importer_flag))
	{
		// Merge all meshes into one big mesh
		aiMatrix4x4 identity;
		parseNode(identity, scene->mRootNode, scene, meshes);
	}

	model = Model(std::move(meshes));
}

void ModelLoader::parseNode(aiMatrix4x4 transform, aiNode *node, const aiScene *scene, std::vector<SubMesh> &meshes)
{
	transform = transform * node->mTransformation;

	for (uint32_t i = 0; i < node->mNumMeshes; i++)
	{
		std::vector<Vertex>   vertices;
		std::vector<uint32_t> indices;
		uint32_t              index_offset = 0;
		aiMesh *              mesh         = scene->mMeshes[node->mMeshes[i]];

		

		parseMesh(transform, mesh, scene, vertices, indices);

		for (auto &submesh : meshes)
		{
			index_offset += static_cast<uint32_t>(submesh.indices.size());
		}

		meshes.emplace_back(std::move(vertices), std::move(indices), index_offset, createScope<material::DisneyPBR>());
	}

	for (uint32_t i = 0; i < node->mNumChildren; i++)
	{
		parseNode(transform, node->mChildren[i], scene, meshes);
	}
}

void ModelLoader::parseMesh(aiMatrix4x4 transform, aiMesh *mesh, const aiScene *scene, std::vector<Vertex> &vertices, std::vector<uint32_t> &indices)
{
	for (uint32_t i = 0; i < mesh->mNumVertices; i++)
	{
		aiVector3D position  = transform * mesh->mVertices[i];
		aiVector3D normal    = mesh->mNormals ? transform * mesh->mNormals[i] : aiVector3D(0.f, 0.f, 0.f);
		aiVector2D texcoords = mesh->mTextureCoords[0] ? aiVector2D(mesh->mTextureCoords[0][i].x, mesh->mTextureCoords[0][i].y) : aiVector2D(0.f, 0.f);
		aiVector3D tangent   = mesh->mTangents ? transform * mesh->mTangents[i] : aiVector3D(0.f, 0.f, 0.f);
		aiVector3D bitangent = mesh->mBitangents ? transform * mesh->mBitangents[i] : aiVector3D(0.f, 0.f, 0.f);

		vertices.emplace_back(to_vector(position), to_vector(texcoords), to_vector(normal), to_vector(tangent), to_vector(bitangent));
	}

	for (uint32_t i = 0; i < mesh->mNumFaces; i++)
	{
		for (uint32_t j = 0; j < 3; j++)
		{
			indices.push_back(mesh->mFaces[i].mIndices[j]);
		}
	}
}
}        // namespace Ilum