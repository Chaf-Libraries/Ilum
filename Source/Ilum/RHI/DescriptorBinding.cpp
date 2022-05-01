//#pragma once
//
//#include <RHI/Buffer.hpp>
//#include <RHI/Texture.hpp>
//
//#include <vector>
//#include <unordered_map>
//
//namespace Ilum
//{
//class ResolveInfo
//{
//  public:
//	void resolve(const std::string &name, Buffer *buffer);
//	void resolve(const std::string &name, Texture *texture);
//	void resolve(const std::string &name, const AccelerationStructure *acceleration_structure);
//	void resolve(const std::string &name, const std::vector<Buffer *> &buffers);
//	void resolve(const std::string &name, const std::vector<Texture *> &images);
//	void resolve(const std::string &name, const std::vector<AccelerationStructureReference> &acceleration_structures);
//
//	const std::unordered_map<std::string, std::vector<Buffer *>>                       &GetBuffers() const;
//	const std::unordered_map<std::string, std::vector<Texture *>>                      &GetImages() const;
//	const std::unordered_map<std::string, std::vector<AccelerationStructureReference>> &GetAccelerationStructures() const;
//
//  private:
//	std::unordered_map<std::string, std::vector<Buffer *>>                       m_buffer_resolves;
//	std::unordered_map<std::string, std::vector<Texture *>>                      m_image_resolves;
//	std::unordered_map<std::string, std::vector<AccelerationStructureReference>> m_acceleration_structure_resolves;
//};
//}