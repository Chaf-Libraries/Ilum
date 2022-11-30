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
using Submesh = Resource<ResourceType::Model>::Submesh;
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
	void ProcessNode(const aiScene *scene, aiNode *node)
	{
	}

	void ProcessBone(const aiScene *assimp_scene, aiMesh *assimp_mesh, std::map<std::string, Bone> &bone_map, Mesh &mesh)
	{
		for (uint32_t bone_index = 0; bone_index < assimp_mesh->mNumBones; bone_index++)
		{
			int32_t     bone_id   = -1;
			std::string bone_name = assimp_mesh->mBones[bone_index]->mName.C_Str();

			if (bone_map.find(bone_name) == bone_map.end())
			{
				Bone bone;
			}
			else
			{
				bone_id = bone_map[bone_name].id;
			}

			auto *  weights    = assimp_mesh->mBones[bone_index]->mWeights;
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

	void ProcessMesh(const aiScene *scene, aiMesh *mesh, std::vector<Mesh> &meshes)
	{
		std::vector<Vertex>   vertices;
		std::vector<uint32_t> indices;

		vertices.reserve(mesh->mNumVertices);
		indices.reserve(mesh->mNumFaces * 3);

		for (uint32_t j = 0; j < mesh->mNumVertices; j++)
		{
			Vertex vertex    = {};
			vertex.position  = glm::vec3(mesh->mVertices[j].x, mesh->mVertices[j].y, mesh->mVertices[j].z);
			vertex.normal    = mesh->mNormals ? glm::vec3(mesh->mNormals[j].x, mesh->mNormals[j].y, mesh->mNormals[j].z) : glm::vec3(0.f);
			vertex.uv0       = mesh->mTextureCoords[0] ? glm::vec2(mesh->mTextureCoords[0][j].x, mesh->mTextureCoords[0][j].y) : glm::vec2(0.f);
			vertex.uv1       = mesh->mTextureCoords[1] ? glm::vec2(mesh->mTextureCoords[1][j].x, mesh->mTextureCoords[1][j].y) : glm::vec2(0.f);
			vertex.tangent   = mesh->mTangents ? glm::vec3(mesh->mTangents[j].x, mesh->mTangents[j].y, mesh->mTangents[j].z) : glm::vec3(0.f);
			vertex.bitangent = mesh->mBitangents ? glm::vec3(mesh->mTangents[j].x, mesh->mTangents[j].y, mesh->mTangents[j].z) : glm::vec3(0.f);
			vertices.emplace_back(std::move(vertex));
		}

		for (uint32_t j = 0; j < mesh->mNumFaces; j++)
		{
			for (uint32_t k = 0; k < 3; k++)
			{
				indices.push_back(mesh->mFaces[j].mIndices[k]);
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

			const size_t max_vertices  = 64;
			const size_t max_triangles = 124;
			const float  cone_weight   = 0.5f;

			std::vector<uint32_t> meshlet_offsets;
			std::vector<uint32_t> meshlet_counts;

			uint32_t meshlet_vertices_offset  = 0;
			uint32_t meshlet_primitive_offset = 0;

			std::vector<Vertex>   model_vertices;
			std::vector<uint32_t> model_indices;
			std::vector<Meshlet>  model_meshlets;
			std::vector<uint32_t> model_meshlet_vertices;
			std::vector<uint32_t> model_meshlet_primitives;

			for (uint32_t i = 0; i < scene->mNumMeshes; i++)
			{
				std::vector<Vertex>   vertices;
				std::vector<uint32_t> indices;

				auto *mesh = scene->mMeshes[i];

				for (uint32_t j = 0; j < mesh->mNumVertices; j++)
				{
					Vertex vertex   = {};
					vertex.position = glm::vec3(mesh->mVertices[j].x, mesh->mVertices[j].y, mesh->mVertices[j].z);
					vertex.normal   = mesh->mNormals ? glm::vec3(mesh->mNormals[j].x, mesh->mNormals[j].y, mesh->mNormals[j].z) : glm::vec3(0.f);
					vertex.uv0      = mesh->mTextureCoords[0] ? glm::vec2(mesh->mTextureCoords[0][j].x, mesh->mTextureCoords[0][j].y) : glm::vec2(0.f);
					vertex.uv1      = mesh->mTextureCoords[1] ? glm::vec2(mesh->mTextureCoords[1][j].x, mesh->mTextureCoords[1][j].y) : glm::vec2(0.f);
					vertex.tangent  = mesh->mTangents ? glm::vec3(mesh->mTangents[j].x, mesh->mTangents[j].y, mesh->mTangents[j].z) : glm::vec3(0.f);
					vertices.emplace_back(std::move(vertex));
				}

				for (uint32_t j = 0; j < mesh->mNumFaces; j++)
				{
					for (uint32_t k = 0; k < 3; k++)
					{
						indices.push_back(mesh->mFaces[j].mIndices[k]);
					}
				}

				// Optimize mesh
				meshopt_optimizeVertexCache(indices.data(), indices.data(), indices.size(), vertices.size());
				meshopt_optimizeOverdraw(indices.data(), indices.data(), indices.size(), &vertices[0].position.x, vertices.size(), sizeof(Vertex), 1.05f);
				meshopt_optimizeVertexFetch(vertices.data(), indices.data(), indices.size(), vertices.data(), vertices.size(), sizeof(Vertex));

				// Generate meshlets
				size_t                       max_meshlets = meshopt_buildMeshletsBound(indices.size(), max_vertices, max_triangles);
				std::vector<meshopt_Meshlet> meshlets(max_meshlets);
				std::vector<uint32_t>        meshlet_vertices(max_meshlets * max_vertices);
				std::vector<uint8_t>         meshlet_triangles(max_meshlets * max_triangles * 3);
				size_t                       meshlet_count = meshopt_buildMeshlets(meshlets.data(), meshlet_vertices.data(), meshlet_triangles.data(), indices.data(),
                                                             indices.size(), &vertices[0].position.x, vertices.size(), sizeof(Vertex), max_vertices, max_triangles, cone_weight);

				// Merge meshlets
				const meshopt_Meshlet &last = meshlets[meshlet_count - 1];
				meshlet_vertices.resize(last.vertex_offset + last.vertex_count);
				meshlet_triangles.resize(last.triangle_offset + ((last.triangle_count * 3 + 3) & ~3));
				meshlets.resize(meshlet_count);

				meshlet_offsets.push_back(meshlet_offsets.empty() ? 0 : meshlet_offsets.back() + meshlet_counts.back());
				meshlet_counts.push_back(static_cast<uint32_t>(meshlet_count));

				// Process meshlets
				std::vector<uint32_t> meshlet_primitives;
				meshlet_primitives.reserve(meshlet_triangles.size() / 3);

				std::vector<meshopt_Bounds> meshlet_bounds;

				for (auto &meshlet : meshlets)
				{
					Meshlet tmp_meshlet = {};

					auto bound = meshopt_computeMeshletBounds(&meshlet_vertices[meshlet.vertex_offset], &meshlet_triangles[meshlet.triangle_offset],
					                                          meshlet.triangle_count, &vertices[0].position.x, vertices.size(), sizeof(Vertex));

					std::memcpy(&tmp_meshlet.center, bound.center, 3 * sizeof(float));
					std::memcpy(&tmp_meshlet.cone_axis, bound.cone_axis, 3 * sizeof(float));

					tmp_meshlet.radius        = bound.radius;
					tmp_meshlet.cone_cutoff   = bound.cone_cutoff;
					tmp_meshlet.vertex_offset = static_cast<uint32_t>(model_vertices.size());
					tmp_meshlet.vertex_count  = meshlet.vertex_count;
					tmp_meshlet.index_offset  = static_cast<uint32_t>(model_indices.size());
					tmp_meshlet.index_count   = meshlet.triangle_count * 3;

					tmp_meshlet.meshlet_vertex_offset    = meshlet_vertices_offset + meshlet.vertex_offset;
					tmp_meshlet.meshlet_primitive_offset = meshlet_primitive_offset;

					for (uint32_t j = 0; j < meshlet.triangle_count; j++)
					{
						uint8_t v0 = meshlet_triangles[meshlet.triangle_offset + j * 3];
						uint8_t v1 = meshlet_triangles[meshlet.triangle_offset + j * 3 + 1];
						uint8_t v2 = meshlet_triangles[meshlet.triangle_offset + j * 3 + 2];
						meshlet_primitives.push_back(PackTriangle(v0, v1, v2));
					}

					model_meshlets.emplace_back(std::move(tmp_meshlet));

					meshlet_primitive_offset += meshlet.triangle_count;
				}

				meshlet_vertices_offset += static_cast<uint32_t>(meshlet_vertices.size());

				model_meshlet_vertices.insert(model_meshlet_vertices.end(), std::make_move_iterator(meshlet_vertices.begin()), std::make_move_iterator(meshlet_vertices.end()));
				model_meshlet_primitives.insert(model_meshlet_primitives.end(), std::make_move_iterator(meshlet_primitives.begin()), std::make_move_iterator(meshlet_primitives.end()));
				model_vertices.insert(model_vertices.end(), std::make_move_iterator(vertices.begin()), std::make_move_iterator(vertices.end()));
				model_indices.insert(model_indices.end(), std::make_move_iterator(indices.begin()), std::make_move_iterator(indices.end()));
			}

			aiMatrix4x4 identity;
			//ParseNode(filename, identity, scene->mRootNode, scene, info, meshlet_offsets, meshlet_counts);
		}

		return info;
	}
};

extern "C"
{
	EXPORT_API AssimpImporter *Create()
	{
		return new AssimpImporter;
	}
}