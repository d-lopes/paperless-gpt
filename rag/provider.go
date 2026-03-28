package rag

import (
	"context"
	"fmt"
)

// Document represents the extracted document information
type Document struct {
	ID            int
	Title         string
	Content       string
	Tags          []string
	DocumentType  string
	Correspondent string
	CreatedDate   string
}

// RagEmbedder defines how to embed text into a vector
type RagEmbedder interface {
	EmbedQuery(ctx context.Context, text string) ([]float32, error)
}

// RagProvider defines the interface for RAG integrations
type RagProvider interface {
	PushDocument(ctx context.Context, doc Document) error
	DeleteDocument(ctx context.Context, documentID int) error
	GetAllDocumentIDs(ctx context.Context) ([]int, error)
}

// NewProvider creates a new RAG provider based on configuration
func NewProvider(providerType string, cfg map[string]string, embedder RagEmbedder) (RagProvider, error) {
	if providerType == "qdrant" {
		return NewQdrantProvider(cfg, embedder)
	} else if providerType == "chromadb" {
		return NewChromaProvider(cfg, embedder)
	}
	return nil, fmt.Errorf("unknown RAG provider: %s", providerType)
}
