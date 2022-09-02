#pragma once

#include <CPP14ParserBaseVisitor.h>

#include "Meta.hpp"

#include <iostream>

namespace Ilum
{
class TreeShapeVisitor : public CPP14ParserBaseVisitor
{
  public:
	virtual std::any visitChildren(antlr4::tree::ParseTree *node) override;
	// Namespace
	virtual std::any visitNamespaceDefinition(CPP14Parser::NamespaceDefinitionContext *ctx) override;
	virtual std::any visitQualifiednamespacespecifier(CPP14Parser::QualifiednamespacespecifierContext *ctx) override;
	virtual std::any visitNestedNameSpecifier(CPP14Parser::NestedNameSpecifierContext *ctx) override;
	// Attribute
	virtual std::any visitAttribute(CPP14Parser::AttributeContext *ctx) override;
	virtual std::any visitAttributeSpecifier(CPP14Parser::AttributeSpecifierContext *ctx) override;
	virtual std::any visitAttributeSpecifierSeq(CPP14Parser::AttributeSpecifierSeqContext *ctx) override;
	// Enumeration
	virtual std::any visitEnumSpecifier(CPP14Parser::EnumSpecifierContext *ctx) override;
	virtual std::any visitEnumeratorList(CPP14Parser::EnumeratorListContext *ctx) override;
	virtual std::any visitEnumeratorDefinition(CPP14Parser::EnumeratorDefinitionContext *ctx) override;
	// Class
	virtual std::any visitAccessSpecifier(CPP14Parser::AccessSpecifierContext *ctx) override;
	virtual std::any visitClassSpecifier(CPP14Parser::ClassSpecifierContext *ctx) override;
	virtual std::any visitMemberdeclaration(CPP14Parser::MemberdeclarationContext *ctx) override;
	virtual std::any visitMemberDeclaratorList(CPP14Parser::MemberDeclaratorListContext *ctx) override;
	virtual std::any visitMemberDeclarator(CPP14Parser::MemberDeclaratorContext *ctx) override;
	virtual std::any visitPointerDeclarator(CPP14Parser::PointerDeclaratorContext *ctx) override;
	virtual std::any visitNoPointerDeclarator(CPP14Parser::NoPointerDeclaratorContext *ctx) override;
	virtual std::any visitBraceOrEqualInitializer(CPP14Parser::BraceOrEqualInitializerContext *ctx) override;
	virtual std::any visitSimpleDeclaration(CPP14Parser::SimpleDeclarationContext *ctx) override;
	virtual std::any visitInitDeclaratorList(CPP14Parser::InitDeclaratorListContext *ctx) override;
	virtual std::any visitInitDeclarator(CPP14Parser::InitDeclaratorContext *ctx) override;
	virtual std::any visitFunctionDefinition(CPP14Parser::FunctionDefinitionContext *ctx) override;

  public:
	const std::vector<Meta::MetaType> &GetMetaTypes() const;

  private:
	std::vector<Meta::AccessSpecifier> m_access_spec_stack;
	std::vector<std::string>           m_namespace_stack;
	std::vector<Meta::MetaType>        m_meta_types;
};

}        // namespace Ilum