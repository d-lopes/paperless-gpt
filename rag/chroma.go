package rag

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"

	"github.com/google/uuid"
	"github.com/tmc/langchaingo/textsplitter"
)

type ChromaProvider struct {
	URL        string
	Collection string
	Client     *http.Client
	Embedder   RagEmbedder
}

func NewChromaProvider(cfg map[string]string, embedder RagEmbedder) (*ChromaProvider, error) {
	url := cfg["RAG_PROVIDER_URL"]
	collection := cfg["RAG_COLLECTION"]
	if url == "" || collection == "" {
		return nil, fmt.Errorf("RAG_PROVIDER_URL and RAG_COLLECTION are required for Chroma RAG integration")
	}

	p := &ChromaProvider{
		URL:        strings.TrimRight(url, "/"),
		Collection: collection,
		Client:     &http.Client{},
		Embedder:   embedder,
	}

	if err := p.ensureCollectionExists(context.Background()); err != nil {
		return nil, fmt.Errorf("failed to ensure Chroma collection exists: %w", err)
	}

	return p, nil
}

func (p *ChromaProvider) ensureCollectionExists(ctx context.Context) error {
	url := fmt.Sprintf("%s/api/v1/collections/%s", p.URL, p.Collection)
	req, _ := http.NewRequestWithContext(ctx, "GET", url, nil)
	resp, err := p.Client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode == 200 {
		return nil // Collection already exists
	}

	createUrl := fmt.Sprintf("%s/api/v1/collections", p.URL)
	createPayload := map[string]interface{}{
		"name": p.Collection,
	}
	jsonStr, _ := json.Marshal(createPayload)
	createReq, _ := http.NewRequestWithContext(ctx, "POST", createUrl, bytes.NewBuffer(jsonStr))
	createReq.Header.Set("Content-Type", "application/json")
	createResp, err := p.Client.Do(createReq)
	if err != nil {
		return err
	}
	defer createResp.Body.Close()

	if createResp.StatusCode < 200 || createResp.StatusCode >= 300 {
		bodyBytes, _ := io.ReadAll(createResp.Body)
		return fmt.Errorf("failed to create collection, status %d: %s", createResp.StatusCode, string(bodyBytes))
	}
	return nil
}

func (p *ChromaProvider) PushDocument(ctx context.Context, doc Document) error {
	if err := p.DeleteDocument(ctx, doc.ID); err != nil {
		fmt.Printf("Warning: failed to delete existing document chunks for ID %d: %v\n", doc.ID, err)
	}

	url := fmt.Sprintf("%s/api/v1/collections/%s/add", p.URL, p.Collection)

	// Determine chunk size
	var chunkSize int
	if limitStr := os.Getenv("RAG_EMBEDDING_TOKEN_LIMIT"); limitStr != "" {
		if val, err := strconv.Atoi(limitStr); err == nil && val > 0 {
			chunkSize = val * 4
		}
	}
	if chunkSize == 0 {
		chunkSize = 2000
	}

	splitter := textsplitter.NewRecursiveCharacter()
	splitter.ChunkSize = chunkSize
	splitter.ChunkOverlap = chunkSize / 10

	chunks, err := splitter.SplitText(doc.Content)
	if err != nil || len(chunks) == 0 {
		chunks = []string{doc.Content}
	}

	var ids []string
	var embeddings [][]float32
	var metadatas []map[string]interface{}
	var documents []string

	for i, chunk := range chunks {
		var vector []float32
		if p.Embedder != nil {
			vector, err = p.Embedder.EmbedQuery(ctx, chunk)
			if err != nil {
				return fmt.Errorf("failed to embed document chunk %d: %w", i, err)
			}
		}

		pointID := uuid.New().String()

		ids = append(ids, pointID)
		metadatas = append(metadatas, map[string]interface{}{
			"document_id":   doc.ID,
			"title":         doc.Title,
			"document_type": doc.DocumentType,
			"correspondent": doc.Correspondent,
			"created_date":  doc.CreatedDate,
			"chunk_index":   i,
			"total_chunks":  len(chunks),
		})

		if p.Embedder != nil {
			embeddings = append(embeddings, vector)
		}
		documents = append(documents, chunk)
	}

	payload := map[string]interface{}{
		"ids":       ids,
		"metadatas": metadatas,
		"documents": documents,
	}

	if p.Embedder != nil {
		payload["embeddings"] = embeddings
	}

	jsonStr, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("failed to marshal chroma payload: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonStr))
	if err != nil {
		return fmt.Errorf("failed to create chroma request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := p.Client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to execute chroma request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("chroma returned status %d: %s", resp.StatusCode, string(bodyBytes))
	}

	return nil
}

func (p *ChromaProvider) DeleteDocument(ctx context.Context, documentID int) error {
	url := fmt.Sprintf("%s/api/v1/collections/%s/delete", p.URL, p.Collection)

	payload := map[string]interface{}{
		"where": map[string]interface{}{
			"document_id": documentID,
		},
	}

	jsonStr, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("failed to marshal chroma delete payload: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonStr))
	if err != nil {
		return fmt.Errorf("failed to create chroma delete request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := p.Client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to execute chroma delete request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("chroma delete returned status %d: %s", resp.StatusCode, string(bodyBytes))
	}

	return nil
}

func (p *ChromaProvider) GetAllDocumentIDs(ctx context.Context) ([]int, error) {
	url := fmt.Sprintf("%s/api/v1/collections/%s/get", p.URL, p.Collection)

	payload := map[string]interface{}{
		"include": []string{"metadatas"},
	}

	jsonStr, _ := json.Marshal(payload)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonStr))
	if err != nil {
		return nil, fmt.Errorf("failed to create chroma get request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := p.Client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to execute chroma get request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("chroma get returned status %d: %s", resp.StatusCode, string(bodyBytes))
	}

	// Read body to string for debugging / parsing
	bodyBytes, _ := io.ReadAll(resp.Body)

	var result struct {
		Metadatas []map[string]interface{} `json:"metadatas"`
	}

	if err := json.Unmarshal(bodyBytes, &result); err != nil {
		return nil, fmt.Errorf("failed to parse chroma get response: %w", err)
	}

	uniqueIDs := make(map[int]struct{})
	for _, m := range result.Metadatas {
		if rawID, ok := m["document_id"]; ok {
			switch v := rawID.(type) {
			case float64:
				uniqueIDs[int(v)] = struct{}{}
			case int:
				uniqueIDs[v] = struct{}{}
			}
		}
	}

	var ids []int
	for id := range uniqueIDs {
		ids = append(ids, id)
	}

	return ids, nil
}
