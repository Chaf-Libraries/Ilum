#include <Resource/Importer.hpp>
#include <Resource/Resource/Model.hpp>

#include <assimp/DefaultLogger.hpp>
#include <assimp/Importer.hpp>
#include <assimp/material.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <meshoptimizer.h>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

using namespace Ilum;

using Meshlet = Resource<ResourceType::Model>::Meshlet;
using Mesh    = Resource<ResourceType::Model>::Mesh;
using Vertex  = Resource<ResourceType::Model>::Vertex;
using Bone    = Resource<ResourceType::Model>::Bone;

inline uint32_t PackTriangle(uint8_t v0, uint8_t v1, uint8_t v2)
{
	return static_cast<uint32_t>(v0) + (static_cast<uint32_t>(v1) << 8) + (static_cast<uint32_t>(v2) << 16);
}

inline glm::mat4 ToMatrix(const aiMatrix4x4 &matrix)
{
	return glm::mat4(
	    matrix.a1, matrix.b1, matrix.c1, matrix.d1,
	    matrix.a2, matrix.b2, matrix.c2, matrix.d2,
	    matrix.a3, matrix.b3, matrix.c3, matrix.d3,
	    matrix.a4, matrix.b4, matrix.c4, matrix.d4);
}

inline glm::mat3 ToMatrix(const aiMatrix3x3 &matrix)
{
	return glm::mat3(
	    matrix.a1, matrix.b1, matrix.c1,
	    matrix.a2, matrix.b2, matrix.c2,
	    matrix.a3, matrix.b3, matrix.c3);
}

inline glm::vec2 ToVector(const aiVector2D &vec)
{
	return glm::vec2(vec.x, vec.y);
}

inline glm::vec3 ToVector(const aiVector3D &vec)
{
	return glm::vec3(vec.x, vec.y, vec.z);
}

inline glm::vec3 ToVector(const aiColor3D &color)
{
	return glm::vec3(color.r, color.g, color.b);
}

inline glm::vec4 ToVector(const aiColor4D &color)
{
	return glm::vec4(color.r, color.g, color.b, color.a);
}

inline glm::quat ToQuaternion(const aiQuaternion &quat)
{
	return glm::quat(quat.x, quat.y, quat.z, quat.w);
}

class AssimpImporter : public Importer<ResourceType::Model>
{
  public:
	struct ModelData
	{
		std::vector<Mesh> meshes;

		std::map<std::string, Bone> bone_map;

		int32_t bone_counter = 0;
	};

	void ProcessMesh(const aiScene *scene, aiMesh *mesh, aiMatrix4x4 transform, ModelData &data)
	{
		Mesh result;

		result.vertices.reserve(mesh->mNumVertices);
		result.indices.reserve(mesh->mNumFaces * 3);

		// Process vertices
		for (uint32_t j = 0; j < mesh->mNumVertices; j++)
		{
			Vertex vertex;
			vertex.position  = glm::vec3(mesh->mVertices[j].x, mesh->mVertices[j].y, mesh->mVertices[j].z);
			vertex.normal    = glm::vec3(mesh->mNormals[j].x, mesh->mNormals[j].y, mesh->mNormals[j].z);
			vertex.bitangent = glm::vec3(mesh->mBitangents[j].x, mesh->mBitangents[j].y, mesh->mBitangents[j].z);
			vertex.tangent   = glm::vec3(mesh->mTangents[j].x, mesh->mTangents[j].y, mesh->mTangents[j].z);
			vertex.texcoord0 = glm::vec2(mesh->mTextureCoords[0][j].x, mesh->mTextureCoords[0][j].y);
			vertex.texcoord1 = mesh->mTextureCoords[1] ? glm::vec2(mesh->mTextureCoords[1][j].x, mesh->mTextureCoords[1][j].y) : glm::vec2(0.f);

			result.vertices.emplace_back(vertex);
		}

		// Process indices
		for (uint32_t j = 0; j < mesh->mNumFaces; j++)
		{
			for (uint32_t k = 0; k < 3; k++)
			{
				result.indices.push_back(mesh->mFaces[j].mIndices[k]);
			}
		}

		// Process skeleton
		{
			if (mesh->mNumBones > 0)
			{
				result.has_skeleton = true;
				for (uint32_t bone_index = 0; bone_index < mesh->mNumBones; bone_index++)
				{
					int32_t bone_id = -1;
					std::string bone_name = mesh->mBones[bone_index]->mName.C_Str();
					if (data.bone_map.find(bone_name) == data.bone_map.end())
					{
						Bone bone = {};
						bone.id   = data.bone_counter;
						bone.offset = ToMatrix(mesh->mBones[bone_index]->mOffsetMatrix);
						data.bone_map[bone_name] = bone;
						bone_id                  = data.bone_counter;
						data.bone_counter++;
					}
					else
					{
						bone_id = data.bone_map.at(bone_name).id;
					}
					auto weights    = mesh->mBones[bone_index]->mWeights;
					uint32_t  weight_num = mesh->mBones[bone_index]->mNumWeights;

					for (uint32_t weight_index = 0; weight_index < weight_num; weight_index++)
					{
						int32_t vertex_id = weights[weight_index].mVertexId;
						float   weight    = weights[weight_index].mWeight;
						for (int i = 0; i < MAX_BONE_INFLUENCE; ++i)
						{
							if (result.vertices[vertex_id].bones[i] < 0)
							{
								result.vertices[vertex_id].weights[i] = weight;
								result.vertices[vertex_id].bones[i]   = bone_id;
								break;
							}
						}
					}
				}
			}
		}

		const size_t max_vertices  = 64;
		const size_t max_triangles = 124;
		const float  cone_weight   = 0.5f;

		// Optimize mesh
		meshopt_optimizeVertexCache(result.indices.data(), result.indices.data(), result.indices.size(), result.vertices.size());
		meshopt_optimizeOverdraw(result.indices.data(), result.indices.data(), result.indices.size(), &result.vertices[0].position.x, result.vertices.size(), sizeof(Vertex), 1.05f);
		meshopt_optimizeVertexFetch(result.vertices.data(), result.indices.data(), result.indices.size(), result.vertices.data(), result.vertices.size(), sizeof(Vertex));

		// Generate meshlets
		size_t                       max_meshlets = meshopt_buildMeshletsBound(result.indices.size(), max_vertices, max_triangles);
		std::vector<meshopt_Meshlet> meshlets(max_meshlets);
		std::vector<uint32_t>        meshlet_vertices(max_meshlets * max_vertices);
		std::vector<uint8_t>         meshlet_triangles(max_meshlets * max_triangles * 3);
		size_t                       meshlet_count = meshopt_buildMeshlets(meshlets.data(), meshlet_vertices.data(), meshlet_triangles.data(), result.indices.data(),
		                                                                   result.indices.size(), &result.vertices[0].position.x, result.vertices.size(), sizeof(Vertex), max_vertices, max_triangles, cone_weight);

		// Merge meshlets
		const meshopt_Meshlet &last = meshlets[meshlet_count - 1];
		meshlet_vertices.resize(last.vertex_offset + last.vertex_count);
		meshlet_triangles.resize(last.triangle_offset + ((last.triangle_count * 3 + 3) & ~3));
		meshlets.resize(meshlet_count);

		result.meshlet_vertices = std::move(meshlet_vertices);

		result.meshlet_primitives.reserve(meshlet_triangles.size() / 3);
		for (uint32_t i = 0; i < meshlet_triangles.size() / 3; i++)
		{
			result.meshlet_primitives.push_back(PackTriangle(meshlet_triangles[3 * i], meshlet_triangles[3 * i + 1], meshlet_triangles[3 * i + 2]));
		}

		data.meshes.emplace_back(std::move(result));
	}

	void ProcessNode(const aiScene *scene, aiNode *node, ModelData &data, aiMatrix4x4 transform = aiMatrix4x4())
	{
		transform = transform * node->mTransformation;

		for (uint32_t i = 0; i < node->mNumMeshes; i++)
		{
			aiMesh *mesh = scene->mMeshes[node->mMeshes[i]];
			ProcessMesh(scene, mesh, transform, data);
		}

		for (uint32_t i = 0; i < node->mNumChildren; i++)
		{
			ProcessNode(scene, node->mChildren[i], data, transform);
		}
	}

	void ProcessBone(const aiScene *assimp_scene, aiMesh *assimp_mesh, std::map<std::string, Bone> &bone_map, Mesh &mesh)
	{
		for (uint32_t bone_index = 0; bone_index < assimp_mesh->mNumBones; bone_index++)
		{
			int32_t     bone_id   = -1;
			std::string bone_name = assimp_mesh->mBones[bone_index]->mName.C_Str();

			if (bone_map.find(bone_name) == bone_map.end())
			{
				// Bone bone;
			}
			else
			{
				bone_id = bone_map[bone_name].id;
			}

			auto   *weights    = assimp_mesh->mBones[bone_index]->mWeights;
			int32_t weight_num = assimp_mesh->mBones[bone_index]->mNumWeights;

			for (int weight_index = 0; weight_index < weight_num; ++weight_index)
			{
				int32_t vertex_id = weights[weight_index].mVertexId;
				float   weight    = weights[weight_index].mWeight;

				for (int32_t i = 0; i < 4; ++i)
				{
					if (mesh.vertices[vertex_id].bones[i] < 0)
					{
						mesh.vertices[vertex_id].weights[i] = weight;
						mesh.vertices[vertex_id].bones[i]   = bone_id;
						break;
					}
				}
			}
		}
	}

	virtual std::unique_ptr<Resource<ResourceType::Model>> Import(const std::string &path, RHIContext *rhi_context) override
	{
		std::string model_name = Path::GetInstance().GetFileName(path, false);

		Assimp::Importer importer;

		if (const aiScene *scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_CalcTangentSpace))
		{
			LOG_INFO("Model {} preprocess finish", path);

			ModelData data;

			aiMatrix4x4 identity;
			ProcessNode(scene, scene->mRootNode, data, identity);
		}

		return nullptr;
	}
};

extern "C"
{
	EXPORT_API AssimpImporter *Create()
	{
		return new AssimpImporter;
	}
}