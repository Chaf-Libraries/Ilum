#include "MetaGenerator.hpp"

namespace Ilum
{
struct DeclSpecifierSeq
{
	std::vector<Meta::DeclSpecifier> decl_specifiers;
	std::vector<Meta::Attribute>          attributes;
};

std::any TreeShapeVisitor::visitChildren(antlr4::tree::ParseTree *node)
{
	std::any result = defaultResult();
	size_t   n      = node->children.size();
	for (size_t i = 0; i < n; i++)
	{
		if (!shouldVisitNextChild(node, result))
		{
			break;
		}

		std::any childResult = node->children[i]->accept(this);
		result               = std::move(childResult);
	}

	return result;
}

std::any TreeShapeVisitor::visitNamespaceDefinition(CPP14Parser::NamespaceDefinitionContext *ctx)
{
	const auto orig_namespace_size = m_namespace_stack.size();

	if (ctx->Identifier())
	{
		m_namespace_stack.push_back(ctx->Identifier()->getText());
	}
	else if (ctx->originalNamespaceName())
	{
		std::vector<std::string> current_namespaces = std::any_cast<std::vector<std::string>>(visitOriginalNamespaceName(ctx->originalNamespaceName()));
		for (auto &namespace_ : current_namespaces)
		{
			m_namespace_stack.push_back(std::move(namespace_));
		}
	}

	if (ctx->declarationseq())
	{
		visit(ctx->declarationseq());
	}

	while (m_namespace_stack.size() > orig_namespace_size)
	{
		m_namespace_stack.pop_back();
	}

	return {};
}

std::any TreeShapeVisitor::visitQualifiednamespacespecifier(CPP14Parser::QualifiednamespacespecifierContext *ctx)
{
	if (ctx->nestedNameSpecifier())
	{
		std::vector<std::string> current_namespaces = std::any_cast<std::vector<std::string>>(visitNestedNameSpecifier(ctx->nestedNameSpecifier()));
		current_namespaces.push_back(ctx->namespaceName()->getText());
		return current_namespaces;
	}
	else
	{
		return std::vector<std::string>{ctx->namespaceName()->getText()};
	}
}

std::any TreeShapeVisitor::visitNestedNameSpecifier(CPP14Parser::NestedNameSpecifierContext *ctx)
{
	return visitChildren(ctx);
	// if (ctx->nestedNameSpecifier())
	//{
	//	std::vector<std::string> current_namespaces = std::any_cast<std::vector<std::string>>(visitNestedNameSpecifier(ctx->nestedNameSpecifier()));
	//	current_namespaces.push_back(ctx->Identifier()->getText());
	//	return current_namespaces;
	// }
	// else
	//{
	//	return std::vector<std::string>{ctx->theTypeName()->getText()};
	// }
}

std::any TreeShapeVisitor::visitAttribute(CPP14Parser::AttributeContext *ctx)
{
	Meta::Attribute attribute = {};
	if (ctx->attributeNamespace())
	{
		attribute._namespace = ctx->attributeNamespace()->getText();
	}
	attribute.name = ctx->Identifier()->getText();
	if (ctx->attributeArgumentClause() && ctx->attributeArgumentClause()->balancedTokenSeq())
	{
		attribute.value = ctx->attributeArgumentClause()->balancedTokenSeq()->getText();
	}

	return attribute;
}

std::any TreeShapeVisitor::visitAttributeSpecifier(CPP14Parser::AttributeSpecifierContext *ctx)
{
	std::vector<Meta::Attribute> attributes;
	if (ctx->attributeList())
	{
		for (auto *ctx_attribute : ctx->attributeList()->attribute())
		{
			Meta::Attribute attribute = std::any_cast<Meta::Attribute>(visit(ctx_attribute));
			attributes.push_back(std::move(attribute));
		}
	}
	return attributes;
}

std::any TreeShapeVisitor::visitAttributeSpecifierSeq(CPP14Parser::AttributeSpecifierSeqContext *ctx)
{
	std::vector<Meta::Attribute> attributes;
	for (auto *ctx_attribute_specifier : ctx->attributeSpecifier())
	{
		std::vector<Meta::Attribute> partial_attributes = std::any_cast<std::vector<Meta::Attribute>>(visit(ctx_attribute_specifier));
		for (auto &attribute : partial_attributes)
		{
			attributes.push_back(std::move(attribute));
		}
	}
	return attributes;
}

std::any TreeShapeVisitor::visitEnumSpecifier(CPP14Parser::EnumSpecifierContext *ctx)
{
	if (!ctx->enumHead()->Identifier())
	{
		return {};
	}

	Meta::MetaType meta = {};

	meta.mode = Meta::MetaType::Mode::Enum;

	meta._namespaces = m_namespace_stack;

	if (ctx->enumHead()->nestedNameSpecifier())
	{
		meta.name = std::any_cast<std::string>(ctx->enumHead()->nestedNameSpecifier());
	}

	meta.name += ctx->enumHead()->Identifier()->getText();

	if (ctx->enumHead()->attributeSpecifierSeq())
	{
		meta.attributes = std::any_cast<std::vector<Meta::Attribute>>(visit(ctx->enumHead()->attributeSpecifierSeq()));
	}

	if (ctx->enumeratorList())
	{
		meta.fields = std::any_cast<std::vector<Meta::Field>>(visit(ctx->enumeratorList()));
	}

	m_meta_types.push_back(std::move(meta));

	return visitChildren(ctx);
}

std::any TreeShapeVisitor::visitEnumeratorList(CPP14Parser::EnumeratorListContext *ctx)
{
	std::vector<Meta::Field> fields;
	for (auto *ctx_enum_definition : ctx->enumeratorDefinition())
	{
		Meta::Field field = std::any_cast<Meta::Field>(visit(ctx_enum_definition));
		fields.push_back(std::move(field));
	}
	return fields;
}

std::any TreeShapeVisitor::visitEnumeratorDefinition(CPP14Parser::EnumeratorDefinitionContext *ctx)
{
	Meta::Field field;
	field.mode      = Meta::Field::Mode::Value;
	field.specifier = Meta::AccessSpecifier::Public;
	field.name      = ctx->enumerator()->getText();
	return field;
}

std::any TreeShapeVisitor::visitAccessSpecifier(CPP14Parser::AccessSpecifierContext *ctx)
{
	if (!m_access_spec_stack.empty())
	{
		if (ctx->Public())
		{
			m_access_spec_stack.back() = Meta::AccessSpecifier::Public;
		}
		else if (ctx->Private())
		{
			m_access_spec_stack.back() = Meta::AccessSpecifier::Private;
		}
		else
		{
			m_access_spec_stack.back() = Meta::AccessSpecifier::Protected;
		}
	}
	return {};
}

std::any TreeShapeVisitor::visitClassSpecifier(CPP14Parser::ClassSpecifierContext *ctx)
{
	if (!m_access_spec_stack.empty() && m_access_spec_stack.back() != Meta::AccessSpecifier::Public)
	{
		return {};
	}

	Meta::MetaType meta = {};

	meta._namespaces = m_namespace_stack;
	meta.name        = ctx->classHead()->classHeadName()->getText();
	m_namespace_stack.push_back(meta.name);

	if (ctx->classHead()->attributeSpecifierSeq())
	{
		meta.attributes = std::any_cast<std::vector<Meta::Attribute>>(visit(ctx->classHead()->attributeSpecifierSeq()));
	}

	// TODO: Template

	if (ctx->classHead()->classKey()->getText() == "class")
	{
		m_access_spec_stack.push_back(Meta::AccessSpecifier::Private);
		meta.mode = Meta::MetaType::Mode::Class;
	}
	else
	{
		m_access_spec_stack.push_back(Meta::AccessSpecifier::Public);
		meta.mode = Meta::MetaType::Mode::Struct;
	}

	// TODO: Base class

	if (ctx->memberSpecification())
	{
		for (auto *child : ctx->memberSpecification()->children)
		{
			auto rst = visit(child);
			if (rst.type().hash_code() == typeid(std::vector<Meta::Field>).hash_code())
			{
				for (auto &field : std::any_cast<std::vector<Meta::Field>>(rst))
				{
					meta.fields.push_back(std::move(field));
				}
			}
		}
	}

	m_meta_types.push_back(std::move(meta));

	m_access_spec_stack.pop_back();
	m_namespace_stack.pop_back();

	return {};
}

std::any TreeShapeVisitor::visitMemberdeclaration(CPP14Parser::MemberdeclarationContext *ctx)
{
	if (ctx->emptyDeclaration() || ctx->aliasDeclaration() || ctx->staticAssertDeclaration() || ctx->usingDeclaration())
	{
		return {};
	}

	if (ctx->declSpecifierSeq())
	{
		return visit(ctx->declSpecifierSeq());
	}
	if (ctx->templateDeclaration())
	{
		return visit(ctx->templateDeclaration());
	}
	if (ctx->functionDefinition())
	{
		//Meta::Field field = std::any_cast<Meta::Field>(visit(ctx->functionDefinition()));
		//return std::vector<Meta::Field>{field};
	}

	std::vector<Meta::Attribute> common_attributes;
	std::vector<Meta::DeclSpecifier> decl_specifiers;

	if (auto* decl_specifier_seq = ctx->declSpecifierSeq())
	{
		// TODO: DeclSpecifierSeq 
	}

	if (ctx->attributeSpecifierSeq())
	{
		std::vector<Meta::Attribute> attributes = std::any_cast<std::vector<Meta::Attribute>>(visit(ctx->attributeSpecifierSeq()));
		for (auto& attribute : attributes)
		{
			common_attributes.push_back(std::move(attribute));
		}
	}

	std::vector<Meta::Field> fields;

	if (ctx->memberDeclaratorList())
	{
		
	}


	return {};
}

std::any TreeShapeVisitor::visitMemberDeclaratorList(CPP14Parser::MemberDeclaratorListContext *ctx)
{
	return visitChildren(ctx);
}

std::any TreeShapeVisitor::visitMemberDeclarator(CPP14Parser::MemberDeclaratorContext *ctx)
{
	Meta::Field field = {};
	if (ctx->declarator())
	{
		field = std::any_cast<Meta::Field>(visit(ctx->declarator()));
	}
	else if (ctx->Identifier())
	{
		field.name = ctx->Identifier()->getText();
		if (ctx->attributeSpecifierSeq())
		{
			field.attributes = std::any_cast<std::vector<Meta::Attribute>>(visit(ctx->attributeSpecifierSeq()));
		}
	}

	if (ctx->braceOrEqualInitializer())
	{
		// TODO: initializer
	}

	return field;
}

const std::vector<Meta::MetaType> &TreeShapeVisitor::GetMetaTypes() const
{
	return m_meta_types;
}
}        // namespace Ilum