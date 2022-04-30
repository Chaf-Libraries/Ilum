#include "ModelLoader.hpp"

#include "File/FileSystem.hpp"

#include "Renderer/Renderer.hpp"

#include "Graphics/Command/CommandBuffer.hpp"

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

	// TODO: Animation
	if (const aiScene *scene = importer.ReadFile(file_path, importer_flag))
	{
		LOG_INFO("Model {} preprocess finish", file_path);
		std::unordered_set<std::string> loaded_textures;

		std::vector<Vertex>   model_vertices;
		std::vector<uint32_t> model_indices;
		std::vector<Meshlet>  model_meshlets;

		const size_t max_vertices  = 64;
		const size_t max_triangles = 124;
		const float  cone_weight   = 0.5f;

		ModelInfo info;

		for (uint32_t i = 0; i < scene->mNumMeshes; i++)
		{
			std::vector<Vertex>   vertices;
			std::vector<uint32_t> indices;

			auto *mesh = scene->mMeshes[i];
			for (uint32_t j = 0; j < mesh->mNumVertices; j++)
			{
				aiVector3D position  = mesh->mVertices[j];
				aiVector3D normal    = mesh->mNormals ? mesh->mNormals[j] : aiVector3D(0.f, 0.f, 0.f);
				aiVector3D tangent   = mesh->mTangents ? mesh->mTangents[j] : aiVector3D(0.f, 0.f, 0.f);
				aiVector2D texcoords = mesh->mTextureCoords[0] ? aiVector2D(mesh->mTextureCoords[0][j].x, mesh->mTextureCoords[0][j].y) : aiVector2D(0.f, 0.f);

				vertices.emplace_back(to_vector(position), to_vector(texcoords), to_vector(normal), to_vector(tangent));
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

			std::vector<meshopt_Bounds> meshlet_bounds;

			uint32_t vertices_count = 0;
			uint32_t indices_count = 0;

			for (auto &meshlet : meshlets)
			{
				Meshlet tmp_meshlet         = {};
				tmp_meshlet.vertices_offset = static_cast<uint32_t>(model_vertices.size());
				tmp_meshlet.vertices_count  = static_cast<uint32_t>(meshlet.vertex_count);
				tmp_meshlet.indices_offset  = static_cast<uint32_t>(model_indices.size());
				tmp_meshlet.indices_count   = static_cast<uint32_t>(meshlet.triangle_count * 3);

				for (uint32_t j = 0; j < meshlet.triangle_count * 3; j++)
				{
					model_indices.push_back(meshlet.vertex_offset + meshlet_triangles[meshlet.triangle_offset + j]);
					indices_count++;
				}

				for (uint32_t j = 0; j < meshlet.vertex_count; j++)
				{
					model_vertices.push_back(vertices[meshlet_vertices[meshlet.vertex_offset + j]]);
					vertices_count++;
				}

				auto bounds = meshopt_computeMeshletBounds(&meshlet_vertices[meshlet.vertex_offset], &meshlet_triangles[meshlet.triangle_offset],
				                                                  meshlet.triangle_count, &vertices[0].position.x, vertices.size(), sizeof(Vertex));
				std::memcpy(glm::value_ptr(tmp_meshlet.center), bounds.center, sizeof(glm::vec3));
				tmp_meshlet.radius = bounds.radius;
				std::memcpy(glm::value_ptr(tmp_meshlet.cone_apex), bounds.cone_apex, sizeof(glm::vec3));
				tmp_meshlet.cone_cutoff = bounds.cone_cutoff;
				std::memcpy(glm::value_ptr(tmp_meshlet.cone_axis), bounds.cone_axis, sizeof(glm::vec3));
			
				model_meshlets.emplace_back(std::move(tmp_meshlet));
			}

			info.meshlets_offsets.push_back(info.meshlets_offsets.empty() ? 0 : info.meshlets_offsets.back() + info.meshlets_counts.back());
			info.meshlets_counts.push_back(static_cast<uint32_t>(meshlet_count));

			info.vertices_offsets.push_back(info.vertices_offsets.empty() ? 0 : info.vertices_offsets.back() + info.vertices_counts.back());
			info.vertices_counts.push_back(vertices_count);

			info.indices_offsets.push_back(info.indices_offsets.empty() ? 0 : info.indices_offsets.back() + info.indices_counts.back());
			info.indices_counts.push_back(indices_count);
		}

		model.vertices_count = static_cast<uint32_t>(model_vertices.size());
		model.indices_count  = static_cast<uint32_t>(model_indices.size());
		model.meshlet_count  = static_cast<uint32_t>(model_meshlets.size());

		{
			Buffer vertex_staging = Buffer(model.vertices_count * sizeof(Vertex), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
			std::memcpy(vertex_staging.map(), model_vertices.data(), model.vertices_count * sizeof(Vertex));

			Buffer index_staging = Buffer(model.indices_count * sizeof(uint32_t), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
			std::memcpy(index_staging.map(), model_indices.data(), model.indices_count * sizeof(uint32_t));

			Buffer meshlet_staging = Buffer(model.meshlet_count * sizeof(Meshlet), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
			std::memcpy(meshlet_staging.map(), model_meshlets.data(), model.meshlet_count * sizeof(Meshlet));

			model.vertices_buffer = Buffer(model.vertices_count * sizeof(Vertex), VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, VMA_MEMORY_USAGE_CPU_TO_GPU);
			model.indices_buffer  = Buffer(model.indices_count * sizeof(uint32_t), VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, VMA_MEMORY_USAGE_CPU_TO_GPU);
			model.meshlets_buffer = Buffer(model.meshlet_count * sizeof(Meshlet), VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

			CommandBuffer cmd_buffer(QueueUsage::Transfer);
			cmd_buffer.begin();
			cmd_buffer.copyBuffer(BufferInfo{vertex_staging}, BufferInfo{model.vertices_buffer}, vertex_staging.getSize());
			cmd_buffer.copyBuffer(BufferInfo{index_staging}, BufferInfo{model.indices_buffer}, index_staging.getSize());
			cmd_buffer.copyBuffer(BufferInfo{meshlet_staging}, BufferInfo{model.meshlets_buffer}, meshlet_staging.getSize());
			cmd_buffer.end();
			cmd_buffer.submitIdle();
		}

		aiMatrix4x4 identity;

		parseNode(file_path, identity, scene->mRootNode, scene, model, info);
	}

	LOG_INFO("Model {} loaded!", file_path);
}

void ModelLoader::parseNode(const std::string &file_path, aiMatrix4x4 transform, aiNode *node, const aiScene *scene, Model &model, const ModelInfo &info)
{
	transform = transform * node->mTransformation;

	for (uint32_t i = 0; i < node->mNumMeshes; i++)
	{
		std::vector<Vertex>   vertices;
		std::vector<uint32_t> indices;

		Material submesh_material;

		uint32_t    index_offset = 0;
		aiMesh     *mesh         = scene->mMeshes[node->mMeshes[i]];
		aiMaterial *material     = scene->mMaterials[mesh->mMaterialIndex];
		parseMaterial(file_path, material, submesh_material);

		SubMesh submesh;
		submesh.name          = mesh->mName.C_Str();
		submesh.index         = node->mMeshes[i];
		submesh.material      = submesh_material;
		submesh.pre_transform = to_matrix(transform);
		submesh.bounding_box  = geometry::BoundingBox(to_vector(mesh->mAABB.mMin), to_vector(mesh->mAABB.mMax));
		model.bounding_box.merge(submesh.bounding_box);

		submesh.vertices_count  = info.vertices_counts[node->mMeshes[i]];
		submesh.vertices_offset = info.vertices_offsets[node->mMeshes[i]];
		submesh.indices_count   = info.indices_counts[node->mMeshes[i]];
		submesh.indices_offset  = info.indices_offsets[node->mMeshes[i]];
		submesh.meshlet_count   = info.meshlets_counts[node->mMeshes[i]];
		submesh.meshlet_offset  = info.meshlets_offsets[node->mMeshes[i]];

		VkAccelerationStructureGeometryKHR geometry_info             = {};
		geometry_info.sType                                          = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
		geometry_info.flags                                          = VK_GEOMETRY_OPAQUE_BIT_KHR;
		geometry_info.geometryType                                   = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
		geometry_info.geometry.triangles.sType                       = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
		geometry_info.geometry.triangles.vertexFormat                = VK_FORMAT_R32G32B32_SFLOAT;
		geometry_info.geometry.triangles.vertexData.deviceAddress    = model.vertices_buffer.getDeviceAddress();
		geometry_info.geometry.triangles.maxVertex                   = submesh.vertices_count;
		geometry_info.geometry.triangles.vertexStride                = sizeof(Vertex);
		geometry_info.geometry.triangles.indexType                   = VK_INDEX_TYPE_UINT32;
		geometry_info.geometry.triangles.indexData.deviceAddress     = model.indices_buffer.getDeviceAddress();
		geometry_info.geometry.triangles.transformData.deviceAddress = 0;
		geometry_info.geometry.triangles.transformData.hostAddress   = nullptr;

		VkAccelerationStructureBuildRangeInfoKHR range_info = {};

		range_info.primitiveCount  = submesh.indices_count / 3;
		range_info.primitiveOffset = (submesh.indices_offset) * sizeof(uint32_t);
		range_info.firstVertex     = submesh.vertices_offset;
		range_info.transformOffset = 0;

		submesh.bottom_level_as.build(geometry_info, range_info);

		model.submeshes.emplace_back(std::move(submesh));
	}

	for (uint32_t i = 0; i < node->mNumChildren; i++)
	{
		parseNode(file_path, transform, node->mChildren[i], scene, model, info);
	}
}

void ModelLoader::parseMaterial(const std::string &file_path, aiMaterial *mesh_material, Material &material)
{
	std::string dictionary = FileSystem::getFileDirectory(file_path);

	aiString path;

	// gltf metal-roughness texture
	aiGetMaterialFloat(mesh_material, AI_MATKEY_METALLIC_FACTOR, &material.metallic);
	aiGetMaterialFloat(mesh_material, AI_MATKEY_ROUGHNESS_FACTOR, &material.roughness);
	aiGetMaterialFloat(mesh_material, AI_MATKEY_BASE_COLOR, glm::value_ptr(material.base_color));
	aiGetMaterialFloat(mesh_material, AI_MATKEY_EMISSIVE_INTENSITY, &material.emissive_intensity);

	if (aiGetMaterialTexture(mesh_material, AI_MATKEY_BASE_COLOR_TEXTURE, &path) != aiReturn_FAILURE)
	{
		material.textures[TextureType::BaseColor] = dictionary + path.C_Str();
		Renderer::instance()->getResourceCache().loadImageAsync(FileSystem::getRelativePath(dictionary + path.C_Str()));
	}
	path.Clear();

	if (aiGetMaterialTexture(mesh_material, aiTextureType_NORMALS, 0, &path) != aiReturn_FAILURE)
	{
		material.textures[TextureType::Normal] = dictionary + path.C_Str();
		Renderer::instance()->getResourceCache().loadImageAsync(FileSystem::getRelativePath(dictionary + path.C_Str()));
	}
	path.Clear();

	if (aiGetMaterialTexture(mesh_material, AI_MATKEY_METALLIC_TEXTURE, &path) != aiReturn_FAILURE)
	{
		material.textures[TextureType::Metallic] = dictionary + path.C_Str();
		Renderer::instance()->getResourceCache().loadImageAsync(FileSystem::getRelativePath(dictionary + path.C_Str()));
	}
	path.Clear();

	if (aiGetMaterialTexture(mesh_material, AI_MATKEY_ROUGHNESS_TEXTURE, &path) != aiReturn_FAILURE)
	{
		material.textures[TextureType::Roughness] = dictionary + path.C_Str();
		Renderer::instance()->getResourceCache().loadImageAsync(FileSystem::getRelativePath(dictionary + path.C_Str()));
	}
	path.Clear();

	if (aiGetMaterialTexture(mesh_material, aiTextureType_EMISSIVE, 0, &path) != aiReturn_FAILURE)
	{
		material.textures[TextureType::Emissive] = dictionary + path.C_Str();
		Renderer::instance()->getResourceCache().loadImageAsync(FileSystem::getRelativePath(dictionary + path.C_Str()));
	}
	path.Clear();

	if (aiGetMaterialTexture(mesh_material, aiTextureType_AMBIENT_OCCLUSION, 0, &path) != aiReturn_FAILURE)
	{
		material.textures[TextureType::AmbientOcclusion] = dictionary + path.C_Str();
		Renderer::instance()->getResourceCache().loadImageAsync(FileSystem::getRelativePath(dictionary + path.C_Str()));
	}
	path.Clear();

	if (FileSystem::getFileExtension(file_path) == ".gltf")
	{
		// For gltf, unknown for roughness & metallic
		if (aiGetMaterialTexture(mesh_material, aiTextureType_UNKNOWN, 0, &path) != aiReturn_FAILURE)
		{
			material.textures[TextureType::Roughness] = dictionary + path.C_Str();
			material.textures[TextureType::Metallic]  = dictionary + path.C_Str();
			Renderer::instance()->getResourceCache().loadImageAsync(FileSystem::getRelativePath(dictionary + path.C_Str()));
		}

		path.Clear();
	}
}
}        // namespace Ilum