#include "Mesh.hpp"

#include <assimp/Importer.hpp>
#include <assimp/pbrmaterial.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <meshoptimizer.h>

#include <deque>
#include <filesystem>

namespace Ilum::Asset
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

Mesh::Mesh(std::pmr::vector<Geo::Vertex> &&vertices, std::pmr::vector<uint32_t> &&indices) :
    m_trimesh(Geo::TriMesh(std::move(vertices), std::move(indices)))
{
	// Add a submesh
	SubMesh submesh;
	submesh.m_vertices_count = static_cast<uint32_t>(m_trimesh.GetVertices().size());
	submesh.m_indices_count  = static_cast<uint32_t>(m_trimesh.GetIndices().size());
	for (auto &vertex : m_trimesh.GetVertices())
	{
		submesh.m_bound.Merge(vertex.position);
	}
	m_submeshes.push_back(submesh);

	m_vertices_count = submesh.GetVerticesCount();
	m_indices_count  = submesh.GetIndicesCount();
	m_bound          = submesh.GetBound();
}

Mesh::Mesh(Mesh &&other) noexcept :
    m_submeshes(other.m_submeshes),
    m_meshlets(other.m_meshlets),
    m_trimesh(std::move(other.m_trimesh)),
    m_bound(std::move(other.m_bound)),
    m_vertices_count(other.m_vertices_count),
    m_indices_count(other.m_indices_count),
    m_vertices_offset(other.m_vertices_offset),
    m_indices_offset(other.m_indices_offset),
    m_hash(other.m_hash)
{
}

Mesh &Mesh::operator=(Mesh &&other) noexcept
{
	m_submeshes       = other.m_submeshes;
	m_meshlets        = other.m_meshlets;
	m_trimesh         = std::move(other.m_trimesh);
	m_bound           = std::move(other.m_bound);
	m_vertices_count  = other.m_vertices_count;
	m_indices_count   = other.m_indices_count;
	m_vertices_offset = other.m_vertices_offset;
	m_indices_offset  = other.m_indices_offset;
	m_hash            = other.m_hash;

	return *this;
}

Mesh Mesh::Create(const std::string &filename)
{
	if (!(std::filesystem::exists(filename) && std::filesystem::is_regular_file(filename)))
	{
		return Mesh();
	}

	Mesh model;

	Assimp::Importer importer;

	const auto importer_flag =
	    aiProcess_FlipUVs |
	    aiProcess_FlipWindingOrder |
	    aiProcess_CalcTangentSpace |
	    aiProcess_GenSmoothNormals |
	    aiProcess_Triangulate |
	    aiProcess_GenBoundingBoxes |
	    aiProcess_JoinIdenticalVertices |
	    aiProcess_GenUVCoords |
	    aiProcess_SortByPType |
	    aiProcess_FindInvalidData |
	    aiProcess_FindInstances |
	    aiProcess_ValidateDataStructure;

	std::pmr::vector<SubMesh> meshes;

	if (const aiScene *scene = importer.ReadFile(filename, importer_flag))
	{
		auto &model_vertices = model.m_trimesh.GetVertices();
		auto &model_indices  = model.m_trimesh.GetIndices();

		const size_t max_vertices  = 64;
		const size_t max_triangles = 124;
		const float  cone_weight   = 1.f;

		std::pmr::vector<uint32_t> meshlet_offsets;
		std::pmr::vector<uint32_t>      meshlet_counts;
		uint32_t                   meshlet_indices_offset = 0;

		for (uint32_t i = 0; i < scene->mNumMeshes; i++)
		{
			std::pmr::vector<Geo::Vertex> vertices;
			std::pmr::vector<uint32_t>    indices;

			auto *mesh = scene->mMeshes[i];
			for (uint32_t j = 0; j < mesh->mNumVertices; j++)
			{
				aiVector3D position  = mesh->mVertices[j];
				aiVector3D normal    = mesh->mNormals ? mesh->mNormals[j] : aiVector3D(0.f, 0.f, 0.f);
				aiVector2D texcoords = mesh->mTextureCoords[0] ? aiVector2D(mesh->mTextureCoords[0][j].x, mesh->mTextureCoords[0][j].y) : aiVector2D(0.f, 0.f);

				vertices.emplace_back(to_vector(position), to_vector(normal), to_vector(texcoords));
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
			meshopt_optimizeOverdraw(indices.data(), indices.data(), indices.size(), &vertices[0].position.x, vertices.size(), sizeof(Geo::Vertex), 1.05f);
			meshopt_optimizeVertexFetch(vertices.data(), indices.data(), indices.size(), vertices.data(), vertices.size(), sizeof(Geo::Vertex));

			// Generate meshlets
			size_t                       max_meshlets = meshopt_buildMeshletsBound(indices.size(), max_vertices, max_triangles);
			std::pmr::vector<meshopt_Meshlet> meshlets(max_meshlets);
			std::pmr::vector<uint32_t>        meshlet_vertices(max_meshlets * max_vertices);
			std::pmr::vector<uint8_t>         meshlet_triangles(max_meshlets * max_triangles * 3);
			size_t                       meshlet_count = meshopt_buildMeshlets(meshlets.data(), meshlet_vertices.data(), meshlet_triangles.data(), indices.data(),
                                                         indices.size(), &vertices[0].position.x, vertices.size(), sizeof(Geo::Vertex), max_vertices, max_triangles, cone_weight);

			// Merge meshlets
			const meshopt_Meshlet &last = meshlets[meshlet_count - 1];
			meshlet_vertices.resize(last.vertex_offset + last.vertex_count);
			meshlet_triangles.resize(last.triangle_offset + ((last.triangle_count * 3 + 3) & ~3));
			meshlets.resize(meshlet_count);

			meshlet_offsets.push_back(meshlet_offsets.empty() ? 0 : meshlet_offsets.back() + meshlet_counts.back());
			meshlet_counts.push_back(static_cast<uint32_t>(meshlet_count));

			// Process meshlets
			std::pmr::vector<uint32_t> meshlet_indices;
			meshlet_indices.reserve(meshlet_triangles.size());

			std::pmr::vector<meshopt_Bounds> meshlet_bounds;

			for (auto &meshlet : meshlets)
			{
				Meshlet tmp_meshlet;
				tmp_meshlet.vertices_offset = static_cast<uint32_t>(model_vertices.size());
				tmp_meshlet.indices_offset  = meshlet_indices_offset;
				tmp_meshlet.indices_count   = static_cast<uint32_t>(meshlet.triangle_count * 3);
				meshlet_indices_offset += tmp_meshlet.indices_count;

				for (uint32_t j = 0; j < meshlet.triangle_count * 3; j++)
				{
					meshlet_indices.push_back(meshlet_vertices[meshlet.vertex_offset + meshlet_triangles[meshlet.triangle_offset + j]]);
				}

				tmp_meshlet.bounds = meshopt_computeMeshletBounds(&meshlet_vertices[meshlet.vertex_offset], &meshlet_triangles[meshlet.triangle_offset],
				                                                  meshlet.triangle_count, &vertices[0].position.x, vertices.size(), sizeof(Geo::Vertex));
				model.m_meshlets.emplace_back(std::move(tmp_meshlet));
			}

			model_vertices.insert(model_vertices.end(), std::make_move_iterator(vertices.begin()), std::make_move_iterator(vertices.end()));
			model_indices.insert(model_indices.end(), std::make_move_iterator(meshlet_indices.begin()), std::make_move_iterator(meshlet_indices.end()));
		}

		model.m_vertices_count = static_cast<uint32_t>(model_vertices.size());
		model.m_indices_count  = static_cast<uint32_t>(model_indices.size());

		std::deque<aiNode *>    nodes;
		std::deque<aiMatrix4x4> transforms;
		nodes.push_back(scene->mRootNode);
		transforms.push_back(aiMatrix4x4());
		while (!nodes.empty())
		{
			auto node = nodes.front();
			nodes.pop_front();

			auto transform = transforms.front();
			transforms.pop_front();
			transform = transform * node->mTransformation;

			for (uint32_t i = 0; i < node->mNumMeshes; i++)
			{
				std::pmr::vector<Geo::Vertex> vertices;
				std::pmr::vector<uint32_t>    indices;

				uint32_t    index_offset = 0;
				aiMesh *    mesh         = scene->mMeshes[node->mMeshes[i]];
				aiMaterial *material     = scene->mMaterials[mesh->mMaterialIndex];

				// Parse material
				Material mesh_material;
				{
					std::string dictionary = filename.substr(0, filename.find_last_of("\\/") + 1);

					aiGetMaterialFloat(material, AI_MATKEY_METALLIC_FACTOR, &mesh_material.metallic_factor);
					aiGetMaterialFloat(material, AI_MATKEY_ROUGHNESS_FACTOR, &mesh_material.roughness_factor);
					aiGetMaterialFloat(material, AI_MATKEY_BASE_COLOR, glm::value_ptr(mesh_material.base_color));
					aiGetMaterialFloat(material, AI_MATKEY_EMISSIVE_INTENSITY, &mesh_material.emissive_intensity);
					aiGetMaterialFloat(material, AI_MATKEY_SPECULAR_FACTOR, &mesh_material.specular_factor);
					aiGetMaterialFloat(material, AI_MATKEY_GLOSSINESS_FACTOR, &mesh_material.glossiness_factor);

					int twosided = 0;
					aiGetMaterialInteger(material, AI_MATKEY_TWOSIDED, &twosided);
					mesh_material.two_sided = static_cast<bool>(twosided);

					int wireframe = 0;
					aiGetMaterialInteger(material, AI_MATKEY_ENABLE_WIREFRAME, &wireframe);
					mesh_material.wireframe = static_cast<bool>(wireframe);

					int opacity = 0;
					aiGetMaterialInteger(material, AI_MATKEY_OPACITY, &opacity);
					mesh_material.opacity = static_cast<bool>(opacity);

#define GET_MATERIAL_TEXTURE(type, uri)                                         \
	{                                                                           \
		aiString path;                                                          \
		if (aiGetMaterialTexture(material, type, 0, &path) != aiReturn_FAILURE) \
		{                                                                       \
			mesh_material.uri = dictionary + path.C_Str();                      \
		}                                                                       \
	}

					// Diffuse
					GET_MATERIAL_TEXTURE(aiTextureType_DIFFUSE, diffuse_map_uri);
					GET_MATERIAL_TEXTURE(aiTextureType_SPECULAR, specular_map_uri);
					GET_MATERIAL_TEXTURE(aiTextureType_AMBIENT, ambient_map_uri);
					GET_MATERIAL_TEXTURE(aiTextureType_EMISSIVE, emissive_map_uri);
					GET_MATERIAL_TEXTURE(aiTextureType_HEIGHT, height_map_uri);
					GET_MATERIAL_TEXTURE(aiTextureType_NORMALS, normal_map_uri);
					GET_MATERIAL_TEXTURE(aiTextureType_SHININESS, shininess_map_uri);
					GET_MATERIAL_TEXTURE(aiTextureType_OPACITY, opacity_map_uri);
					GET_MATERIAL_TEXTURE(aiTextureType_DISPLACEMENT, displacement_map_uri);
					GET_MATERIAL_TEXTURE(aiTextureType_AMBIENT_OCCLUSION, ao_map_uri);
					GET_MATERIAL_TEXTURE(aiTextureType_BASE_COLOR, base_color_map_uri);
					GET_MATERIAL_TEXTURE(aiTextureType_METALNESS, metallic_map_uri);
					GET_MATERIAL_TEXTURE(aiTextureType_DIFFUSE_ROUGHNESS, roughness_map_uri);

					if (std::filesystem::path(filename).extension().generic_string() == ".gltf")
					{
						aiString path;
						if (aiGetMaterialTexture(material, aiTextureType_UNKNOWN, 0, &path) != aiReturn_FAILURE)
						{
							mesh_material.roughness_map_uri = dictionary + path.C_Str();
							mesh_material.metallic_map_uri  = dictionary + path.C_Str();
						}
					}
				}

				// Assign submesh
				SubMesh submesh;
				submesh.m_index          = node->mMeshes[i];
				submesh.m_material       = mesh_material;
				submesh.m_pre_transform  = to_matrix(transform);
				submesh.m_bound          = Geo::Bound(to_vector(mesh->mAABB.mMin), to_vector(mesh->mAABB.mMax));
				submesh.m_vertices_count = mesh->mNumVertices;
				submesh.m_indices_count  = mesh->mNumFaces * 3;

				uint32_t vertices_offset = 0;
				uint32_t indices_offset  = 0;
				for (uint32_t j = 0; j < node->mMeshes[i]; j++)
				{
					vertices_offset += scene->mMeshes[j]->mNumVertices;
					indices_offset += scene->mMeshes[j]->mNumFaces * 3;
				}
				submesh.m_vertices_offset = vertices_offset;
				submesh.m_indices_offset  = indices_offset;
				submesh.m_meshlet_count   = meshlet_counts[node->mMeshes[i]];
				submesh.m_meshlet_offset  = meshlet_offsets[node->mMeshes[i]];

				model.m_bound.Merge(submesh.m_bound);
				model.m_submeshes.emplace_back(std::move(submesh));
			}

			for (uint32_t i = 0; i < node->mNumChildren; i++)
			{
				nodes.push_back(node->mChildren[i]);
				transforms.push_back(transform);
			}
		}
	}

	return model;
}

Mesh Mesh::Create(std::pmr::vector<Geo::Vertex> &&vertices, std::pmr::vector<uint32_t> &&indices)
{
	return Mesh(std::move(vertices), std::move(indices));
}

const std::pmr::vector<SubMesh> &Mesh::GetSubMesh() const
{
	return m_submeshes;
}

const std::pmr::vector<Meshlet> &Mesh::GetMeshlet() const
{
	return m_meshlets;
}

const Geo::TriMesh &Mesh::GetTriMesh() const
{
	return m_trimesh;
}

const Geo::Bound &Mesh::GetBound() const
{
	return m_bound;
}

uint32_t Mesh::GetVerticesCount() const
{
	return m_vertices_count;
}

uint32_t Mesh::GetIndicesCount() const
{
	return uint32_t();
}

uint32_t Mesh::GetVerticesOffset() const
{
	return m_vertices_offset;
}

uint32_t Mesh::GetIndicesOffset() const
{
	return m_vertices_offset;
}

uint32_t Mesh::GetHashValue() const
{
	return m_indices_offset;
}

void Mesh::SetVerticesOffset(uint32_t offset)
{
	m_vertices_offset = offset;
}

void Mesh::SetIndicesOffset(uint32_t offset)
{
	m_indices_count = offset;
}

void Mesh::Save(const std::string &filename)
{
}
}        // namespace Ilum::Asset