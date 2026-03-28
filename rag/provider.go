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

// Embedder defines how to embed text into a vector
type Embedder interface {
	EmbedQuery(ctx context.Context, text string) ([]float32, error)
}

// Provider defines the interface for RAG integrations
type Provider interface {
	PushDocument(ctx context.Context, doc Document) error
}

// NewProvider creates a new RAG provider based on configuration
func NewProvider(providerType string, cfg map[string]string, embedder Embedder) (Provider, error) {
	if providerType == "qdrant" {
		return NewQdrantProvider(cfg, embedder)
	}
	return nil, fmt.Errorf("unknown RAG provider: %s", providerType)
}
