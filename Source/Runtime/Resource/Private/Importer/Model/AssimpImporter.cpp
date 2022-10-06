#include "AssimpImporter.hpp"

#include <Core/Path.hpp>

#include <assimp/DefaultLogger.hpp>
#include <assimp/Importer.hpp>
#include <assimp/material.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <meshoptimizer.h>

namespace Ilum
{
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

inline void ParseNode(const std::string &file_path, aiMatrix4x4 transform, aiNode *node, const aiScene *scene, ModelImportInfo &info, std::vector<uint32_t> &meshlet_offsets, std::vector<uint32_t> &meshlet_counts)
{
	transform = transform * node->mTransformation;

	for (uint32_t i = 0; i < node->mNumMeshes; i++)
	{
		std::vector<Vertex>   vertices;
		std::vector<uint32_t> indices;

		// Material submesh_material;

		uint32_t index_offset = 0;
		aiMesh  *mesh         = scene->mMeshes[node->mMeshes[i]];
		// aiMaterial *material     = scene->mMaterials[mesh->mMaterialIndex];
		// parseMaterial(file_path, material, submesh_material);

		Submesh submesh;
		submesh.name  = mesh->mName.C_Str();
		submesh.index = node->mMeshes[i];
		// submesh.material      = submesh_material;
		submesh.pre_transform = ToMatrix(transform);
		submesh.aabb          = AABB(ToVector(mesh->mAABB.mMin), ToVector(mesh->mAABB.mMax));
		info.aabb.Merge(submesh.aabb);

		submesh.vertices_count = mesh->mNumVertices;
		submesh.indices_count  = mesh->mNumFaces * 3;

		uint32_t vertices_offset = 0;
		uint32_t indices_offset  = 0;
		for (uint32_t j = 0; j < node->mMeshes[i]; j++)
		{
			vertices_offset += scene->mMeshes[j]->mNumVertices;
			indices_offset += scene->mMeshes[j]->mNumFaces * 3;
		}
		submesh.vertices_offset = vertices_offset;
		submesh.indices_offset  = indices_offset;
		submesh.meshlet_count   = meshlet_counts[node->mMeshes[i]];
		submesh.meshlet_offset  = meshlet_offsets[node->mMeshes[i]];

		info.submeshes.emplace_back(std::move(submesh));
	}

	for (uint32_t i = 0; i < node->mNumChildren; i++)
	{
		ParseNode(file_path, transform, node->mChildren[i], scene, info, meshlet_offsets, meshlet_counts);
	}
}

ModelImportInfo AssimpImporter::ImportImpl(const std::string &filename)
{
	ModelImportInfo info;
	info.name = Path::GetInstance().GetFileName(filename, false);

	Assimp::Importer importer;

	const auto importer_flag =
	    aiProcess_FlipUVs |
	    aiProcess_FlipWindingOrder |
	    aiProcess_JoinIdenticalVertices |
	    aiProcess_GenSmoothNormals |
	    aiProcess_Triangulate |
	    aiProcess_GenBoundingBoxes |
	    aiProcess_OptimizeGraph |
	    aiProcess_OptimizeMeshes |
	    aiProcess_GenUVCoords |
	    aiProcess_SortByPType |
	    aiProcess_FindInvalidData |
	    aiProcess_FindInstances |
	    aiProcess_ValidateDataStructure;

	if (const aiScene *scene = importer.ReadFile(filename, importer_flag))
	{
		LOG_INFO("Model {} preprocess finish", filename);

		const size_t max_vertices  = 64;
		const size_t max_triangles = 124;
		const float  cone_weight   = 0.5f;

		std::vector<uint32_t> meshlet_offsets;
		std::vector<uint32_t> meshlet_counts;

		uint32_t meshlet_vertices_offset  = 0;
		uint32_t meshlet_primitive_offset = 0;

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
				vertex.texcoord = mesh->mTextureCoords[0] ? glm::vec2(mesh->mTextureCoords[0][j].x, mesh->mTextureCoords[0][j].y) : glm::vec2(0.f);
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
				Meshlet tmp_meshlet         = {};
				tmp_meshlet.vertices_offset = static_cast<uint32_t>(info.vertices.size());
				tmp_meshlet.vertices_count  = meshlet.vertex_count;
				tmp_meshlet.indices_offset  = static_cast<uint32_t>(info.indices.size());
				tmp_meshlet.indices_count   = meshlet.triangle_count * 3;

				tmp_meshlet.meshlet_vertices_offset  = meshlet_vertices_offset + meshlet.vertex_offset;
				tmp_meshlet.meshlet_primitive_offset = meshlet_primitive_offset + meshlet.triangle_offset / 3;

				for (uint32_t j = 0; j < meshlet.triangle_count * 3; j += 3)
				{
					uint8_t v0 = meshlet_vertices[meshlet.vertex_offset + meshlet_triangles[meshlet.triangle_offset + j]];
					uint8_t v1 = meshlet_vertices[meshlet.vertex_offset + meshlet_triangles[meshlet.triangle_offset + j + 1]];
					uint8_t v2 = meshlet_vertices[meshlet.vertex_offset + meshlet_triangles[meshlet.triangle_offset + j + 2]];
					meshlet_primitives.push_back(PackTriangle(v0, v1, v2));
				}

				auto bound = meshopt_computeMeshletBounds(&meshlet_vertices[meshlet.vertex_offset], &meshlet_triangles[meshlet.triangle_offset],
				                                          meshlet.triangle_count, &vertices[0].position.x, vertices.size(), sizeof(Vertex));
				std::memcpy(&tmp_meshlet.bound, &bound, sizeof(tmp_meshlet.bound));
				info.meshlets.emplace_back(std::move(tmp_meshlet));
			}

			meshlet_vertices_offset += static_cast<uint32_t>(meshlet_vertices.size());
			meshlet_primitive_offset += static_cast<uint32_t>(meshlet_triangles.size()) / 3;

			info.meshlet_vertices.insert(info.meshlet_vertices.end(), std::make_move_iterator(meshlet_vertices.begin()), std::make_move_iterator(meshlet_vertices.end()));
			info.meshlet_primitives.insert(info.meshlet_primitives.end(), std::make_move_iterator(meshlet_primitives.begin()), std::make_move_iterator(meshlet_primitives.end()));
			info.vertices.insert(info.vertices.end(), std::make_move_iterator(vertices.begin()), std::make_move_iterator(vertices.end()));
			info.indices.insert(info.indices.end(), std::make_move_iterator(indices.begin()), std::make_move_iterator(indices.end()));
		}

		aiMatrix4x4 identity;
		ParseNode(filename, identity, scene->mRootNode, scene, info, meshlet_offsets, meshlet_counts);
	}

	return info;
}
}        // namespace Ilum