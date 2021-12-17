#include "ModelLoader.hpp"

#include "File/FileSystem.hpp"

#include "Renderer/Renderer.hpp"

#include <assimp/DefaultLogger.hpp>
#include <assimp/Importer.hpp>
#include <assimp/pbrmaterial.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <meshoptimizer.h>

namespace Ilum
{
inline glm::mat4 to_matrix(const aiMatrix4x4 &matrix)
{
	return glm::mat4(
	    matrix.a1, matrix.b1, matrix.c1, matrix.d1,
	    matrix.a2, matrix.b2, matrix.c2, matrix.d2,
	    matrix.a3, matrix.b3, matrix.c3, matrix.d3,
	    matrix.a4, matrix.b4, matrix.c4, matrix.d4);
}

inline glm::mat3 to_matrix(const aiMatrix3x3 &matrix)
{
	return glm::mat3(
	    matrix.a1, matrix.b1, matrix.c1,
	    matrix.a2, matrix.b2, matrix.c2,
	    matrix.a3, matrix.b3, matrix.c3);
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

	std::vector<SubMesh>     meshes;
	std::vector<std::string> materials;

	const auto importer_flag =
	    aiProcess_FlipUVs |
	    aiProcess_FlipWindingOrder |
	    aiProcess_CalcTangentSpace |
	    aiProcess_GenSmoothNormals |
	    aiProcess_Triangulate |
	    aiProcess_GenBoundingBoxes |
	    aiProcess_GenUVCoords |
	    aiProcess_SortByPType |
	    aiProcess_FindInvalidData |
	    aiProcess_FindInstances |
	    aiProcess_ValidateDataStructure;

	// TODO: Animation
	if (const aiScene *scene = importer.ReadFile(file_path, importer_flag))
	{
		LOG_INFO("Model {} preprocess finish", file_path);
		std::unordered_set<std::string> loaded_textures;

		model.mesh.vertices.clear();
		model.mesh.indices.clear();
		model.meshlets.clear();

		const size_t max_vertices  = 64;
		const size_t max_triangles = 124;
		const float  cone_weight   = 1.f;

		std::vector<uint32_t> meshlet_offsets;
		std::vector<uint32_t> meshlet_counts;
		uint32_t              meshlet_indices_offset = 0;

		for (uint32_t i = 0; i < scene->mNumMeshes; i++)
		{
			std::vector<Vertex>   vertices;
			std::vector<uint32_t> indices;

			auto *mesh = scene->mMeshes[i];
			for (uint32_t j = 0; j < mesh->mNumVertices; j++)
			{
				aiVector3D position  = mesh->mVertices[j];
				aiVector3D normal    = mesh->mNormals ? mesh->mNormals[j] : aiVector3D(0.f, 0.f, 0.f);
				aiVector2D texcoords = mesh->mTextureCoords[0] ? aiVector2D(mesh->mTextureCoords[0][j].x, mesh->mTextureCoords[0][j].y) : aiVector2D(0.f, 0.f);
				aiVector3D tangent   = mesh->mTangents ? mesh->mTangents[j] : aiVector3D(0.f, 0.f, 0.f);
				aiVector3D bitangent = mesh->mBitangents ? mesh->mBitangents[j] : aiVector3D(0.f, 0.f, 0.f);

				vertices.emplace_back(to_vector(position), to_vector(texcoords), to_vector(normal), to_vector(tangent), to_vector(bitangent));
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
			std::vector<uint32_t> meshlet_indices;
			meshlet_indices.reserve(meshlet_triangles.size());

			std::vector<meshopt_Bounds> meshlet_bounds;

			for (auto &meshlet : meshlets)
			{
				Meshlet tmp_meshlet;
				tmp_meshlet.vertices_offset = static_cast<uint32_t>(model.mesh.vertices.size());
				tmp_meshlet.indices_offset = meshlet_indices_offset;
				tmp_meshlet.indices_count  = static_cast<uint32_t>(meshlet.triangle_count * 3);
				meshlet_indices_offset += tmp_meshlet.indices_count;

				for (uint32_t j = 0; j < meshlet.triangle_count * 3; j++)
				{
					meshlet_indices.push_back(meshlet_vertices[meshlet.vertex_offset + meshlet_triangles[meshlet.triangle_offset + j]]);
				}

				tmp_meshlet.bounds = meshopt_computeMeshletBounds(&meshlet_vertices[meshlet.vertex_offset], &meshlet_triangles[meshlet.triangle_offset],
				                                                  meshlet.triangle_count, &vertices[0].position.x, vertices.size(), sizeof(Vertex));
				model.meshlets.emplace_back(std::move(tmp_meshlet));
			}

			model.mesh.vertices.insert(model.mesh.vertices.end(), std::make_move_iterator(vertices.begin()), std::make_move_iterator(vertices.end()));
			model.mesh.indices.insert(model.mesh.indices.end(), std::make_move_iterator(meshlet_indices.begin()), std::make_move_iterator(meshlet_indices.end()));
		}

		model.vertices_count = static_cast<uint32_t>(model.mesh.vertices.size());
		model.indices_count  = static_cast<uint32_t>(model.mesh.indices.size());

		aiMatrix4x4 identity;
		parseNode(file_path, identity, scene->mRootNode, scene, model, meshlet_offsets, meshlet_counts);
	}

	LOG_INFO("Model {} loaded!", file_path);
}

void ModelLoader::parseNode(const std::string &file_path, aiMatrix4x4 transform, aiNode *node, const aiScene *scene, Model &model, std::vector<uint32_t> &meshlet_offsets, std::vector<uint32_t> &meshlet_counts)
{
	transform = transform * node->mTransformation;

	for (uint32_t i = 0; i < node->mNumMeshes; i++)
	{
		std::vector<Vertex>   vertices;
		std::vector<uint32_t> indices;

		auto pbr = createScope<material::DisneyPBR>();

		uint32_t    index_offset = 0;
		aiMesh *    mesh         = scene->mMeshes[node->mMeshes[i]];
		aiMaterial *material     = scene->mMaterials[mesh->mMaterialIndex];

		parseMaterial(file_path, material, pbr);

		SubMesh submesh;
		submesh.index         = node->mMeshes[i];
		submesh.material      = *pbr;
		submesh.pre_transform = to_matrix(transform);
		submesh.bounding_box  = geometry::BoundingBox(to_vector(mesh->mAABB.mMin), to_vector(mesh->mAABB.mMax));
		model.bounding_box.merge(submesh.bounding_box);

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

		model.submeshes.emplace_back(std::move(submesh));
	}

	for (uint32_t i = 0; i < node->mNumChildren; i++)
	{
		parseNode(file_path, transform, node->mChildren[i], scene, model, meshlet_offsets, meshlet_counts);
	}
}

void ModelLoader::parseMaterial(const std::string &file_path, aiMaterial *mesh_material, scope<material::DisneyPBR> &material)
{
	std::string dictionary = FileSystem::getFileDirectory(file_path);

	aiString path;

	// gltf metal-roughness texture
	aiGetMaterialFloat(mesh_material, AI_MATKEY_METALLIC_FACTOR, &material->metallic_factor);
	aiGetMaterialFloat(mesh_material, AI_MATKEY_ROUGHNESS_FACTOR, &material->roughness_factor);
	aiGetMaterialFloat(mesh_material, AI_MATKEY_BASE_COLOR, glm::value_ptr(material->base_color));
	aiGetMaterialFloat(mesh_material, AI_MATKEY_EMISSIVE_INTENSITY, &material->emissive_intensity);

	if (aiGetMaterialTexture(mesh_material, AI_MATKEY_BASE_COLOR_TEXTURE, &path) != aiReturn_FAILURE)
	{
		material->albedo_map = dictionary + path.C_Str();
		Renderer::instance()->getResourceCache().loadImageAsync(FileSystem::getRelativePath(dictionary + path.C_Str()));
	}
	path.Clear();

	if (aiGetMaterialTexture(mesh_material, aiTextureType_NORMALS, 0, &path) != aiReturn_FAILURE)
	{
		material->normal_map = dictionary + path.C_Str();
		Renderer::instance()->getResourceCache().loadImageAsync(FileSystem::getRelativePath(dictionary + path.C_Str()));
	}
	path.Clear();

	if (aiGetMaterialTexture(mesh_material, AI_MATKEY_METALLIC_TEXTURE, &path) != aiReturn_FAILURE)
	{
		material->metallic_map = dictionary + path.C_Str();
		Renderer::instance()->getResourceCache().loadImageAsync(FileSystem::getRelativePath(dictionary + path.C_Str()));
	}
	path.Clear();

	if (aiGetMaterialTexture(mesh_material, AI_MATKEY_ROUGHNESS_TEXTURE, &path) != aiReturn_FAILURE)
	{
		material->roughness_map = dictionary + path.C_Str();
		Renderer::instance()->getResourceCache().loadImageAsync(FileSystem::getRelativePath(dictionary + path.C_Str()));
	}
	path.Clear();

	if (aiGetMaterialTexture(mesh_material, aiTextureType_EMISSIVE, 0, &path) != aiReturn_FAILURE)
	{
		material->emissive_map = dictionary + path.C_Str();
		Renderer::instance()->getResourceCache().loadImageAsync(FileSystem::getRelativePath(dictionary + path.C_Str()));
	}
	path.Clear();

	if (aiGetMaterialTexture(mesh_material, aiTextureType_AMBIENT_OCCLUSION, 0, &path) != aiReturn_FAILURE)
	{
		material->ao_map = dictionary + path.C_Str();
		Renderer::instance()->getResourceCache().loadImageAsync(FileSystem::getRelativePath(dictionary + path.C_Str()));
	}
	path.Clear();

	if (FileSystem::getFileExtension(file_path) == ".gltf")
	{
		// For gltf, unknown for roughness & metallic
		if (aiGetMaterialTexture(mesh_material, aiTextureType_UNKNOWN, 0, &path) != aiReturn_FAILURE)
		{
			material->roughness_map = dictionary + path.C_Str();
			material->metallic_map  = dictionary + path.C_Str();
			Renderer::instance()->getResourceCache().loadImageAsync(FileSystem::getRelativePath(dictionary + path.C_Str()));
		}

		path.Clear();
	}
}
}        // namespace Ilum