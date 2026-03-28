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

type QdrantProvider struct {
	URL        string
	APIKey     string
	Collection string
	Client     *http.Client
	Embedder   RagEmbedder
}

func NewQdrantProvider(cfg map[string]string, embedder RagEmbedder) (*QdrantProvider, error) {
	url := cfg["RAG_PROVIDER_URL"]
	apiKey := cfg["RAG_API_KEY"]
	collection := cfg["RAG_COLLECTION"]
	if url == "" || collection == "" {
		return nil, fmt.Errorf("RAG_PROVIDER_URL and RAG_COLLECTION are required for Qdrant RAG integration")
	}

	p := &QdrantProvider{
		URL:        strings.TrimRight(url, "/"),
		APIKey:     apiKey,
		Collection: collection,
		Client:     &http.Client{},
		Embedder:   embedder,
	}

	// Ensure collection exists
	if err := p.ensureCollectionExists(context.Background()); err != nil {
		return nil, fmt.Errorf("failed to ensure Qdrant collection exists: %w", err)
	}

	return p, nil
}

func (p *QdrantProvider) ensureCollectionExists(ctx context.Context) error {
	url := fmt.Sprintf("%s/collections/%s", p.URL, p.Collection)
	req, _ := http.NewRequestWithContext(ctx, "GET", url, nil)
	if p.APIKey != "" {
		req.Header.Set("api-key", p.APIKey)
	}
	resp, err := p.Client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode == 200 {
		return nil // Collection already exists
	}

	// Create collection
	vectorSize := 1536 // Default for typical OpenAI models
	if p.Embedder != nil {
		dummy, err := p.Embedder.EmbedQuery(ctx, "test")
		if err == nil {
			vectorSize = len(dummy)
		} else {
			fmt.Printf("Warning: failed to get vector size from embedder: %v\n", err)
		}
	}

	createUrl := fmt.Sprintf("%s/collections/%s", p.URL, p.Collection)
	createPayload := map[string]interface{}{
		"vectors": map[string]interface{}{
			"size":     vectorSize,
			"distance": "Cosine",
		},
	}
	jsonStr, _ := json.Marshal(createPayload)
	createReq, _ := http.NewRequestWithContext(ctx, "PUT", createUrl, bytes.NewBuffer(jsonStr))
	createReq.Header.Set("Content-Type", "application/json")
	if p.APIKey != "" {
		createReq.Header.Set("api-key", p.APIKey)
	}
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

func (p *QdrantProvider) PushDocument(ctx context.Context, doc Document) error {
	// First delete any existing points to prevent duplication
	if err := p.DeleteDocument(ctx, doc.ID); err != nil {
		fmt.Printf("Warning: failed to delete existing document chunks for ID %d: %v\n", doc.ID, err)
	}

	url := fmt.Sprintf("%s/collections/%s/points?wait=true", p.URL, p.Collection)

	var chunkSize int
	if limitStr := os.Getenv("RAG_EMBEDDING_TOKEN_LIMIT"); limitStr != "" {
		if val, err := strconv.Atoi(limitStr); err == nil && val > 0 {
			chunkSize = val * 4 // roughly 4 chars per token
		}
	}
	if chunkSize == 0 {
		chunkSize = 2000 // default fallback
	}

	splitter := textsplitter.NewRecursiveCharacter()
	splitter.ChunkSize = chunkSize
	splitter.ChunkOverlap = chunkSize / 10

	chunks, err := splitter.SplitText(doc.Content)
	if err != nil || len(chunks) == 0 {
		chunks = []string{doc.Content}
	}

	var points []map[string]interface{}

	for i, chunk := range chunks {
		var vector []float32
		if p.Embedder != nil {
			vector, err = p.Embedder.EmbedQuery(ctx, chunk)
			if err != nil {
				return fmt.Errorf("failed to embed document chunk %d: %w", i, err)
			}
		}

		// Use UUID instead of integer since one document can have multiple chunks
		pointID := uuid.New().String()

		point := map[string]interface{}{
			"id": pointID,
			"payload": map[string]interface{}{
				"document_id":   doc.ID,
				"title":         doc.Title,
				"content":       chunk,
				"tags":          doc.Tags,
				"document_type": doc.DocumentType,
				"correspondent": doc.Correspondent,
				"created_date":  doc.CreatedDate,
				"chunk_index":   i,
				"total_chunks":  len(chunks),
			},
		}

		if p.Embedder != nil {
			point["vector"] = vector
		} else {
			point["vector"] = map[string]interface{}{}
		}

		points = append(points, point)
	}

	payload := map[string]interface{}{
		"points": points,
	}

	jsonStr, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("failed to marshal qdrant payload: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "PUT", url, bytes.NewBuffer(jsonStr))
	if err != nil {
		return fmt.Errorf("failed to create qdrant request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if p.APIKey != "" {
		req.Header.Set("api-key", p.APIKey)
	}

	resp, err := p.Client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to execute qdrant request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("qdrant returned status %d: %s", resp.StatusCode, string(bodyBytes))
	}

	return nil
}

func (p *QdrantProvider) DeleteDocument(ctx context.Context, documentID int) error {
	url := fmt.Sprintf("%s/collections/%s/points/delete?wait=true", p.URL, p.Collection)

	payload := map[string]interface{}{
		"filter": map[string]interface{}{
			"must": []map[string]interface{}{
				{
					"key": "document_id",
					"match": map[string]interface{}{
						"value": documentID,
					},
				},
			},
		},
	}

	jsonStr, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("failed to marshal qdrant delete payload: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonStr))
	if err != nil {
		return fmt.Errorf("failed to create qdrant delete request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if p.APIKey != "" {
		req.Header.Set("api-key", p.APIKey)
	}

	resp, err := p.Client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to execute qdrant delete request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("qdrant delete returned status %d: %s", resp.StatusCode, string(bodyBytes))
	}

	return nil
}

func (p *QdrantProvider) GetAllDocumentIDs(ctx context.Context) ([]int, error) {
	url := fmt.Sprintf("%s/collections/%s/points/scroll", p.URL, p.Collection)

	uniqueIDs := make(map[int]struct{})
	var nextOffset interface{} = nil
	limit := 1000

	for {
		payload := map[string]interface{}{
			"limit":        limit,
			"with_payload": []string{"document_id"},
			"with_vector":  false,
		}
		if nextOffset != nil {
			payload["offset"] = nextOffset
		}

		jsonStr, _ := json.Marshal(payload)

		req, _ := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonStr))
		req.Header.Set("Content-Type", "application/json")
		if p.APIKey != "" {
			req.Header.Set("api-key", p.APIKey)
		}

		resp, err := p.Client.Do(req)
		if err != nil {
			return nil, fmt.Errorf("failed to scroll qdrant: %w", err)
		}

		bodyBytes, _ := io.ReadAll(resp.Body)
		resp.Body.Close()

		if resp.StatusCode != 200 {
			return nil, fmt.Errorf("qdrant scroll failed: status %d %s", resp.StatusCode, string(bodyBytes))
		}

		var result struct {
			Result struct {
				Points []struct {
					Payload struct {
						DocumentID float64 `json:"document_id"`
					} `json:"payload"`
				} `json:"points"`
				NextPageOffset interface{} `json:"next_page_offset"`
			} `json:"result"`
		}

		if err := json.Unmarshal(bodyBytes, &result); err != nil {
			return nil, fmt.Errorf("failed to parse qdrant scroll response: %w", err)
		}

		for _, pt := range result.Result.Points {
			id := int(pt.Payload.DocumentID)
			if id > 0 {
				uniqueIDs[id] = struct{}{}
			}
		}

		if result.Result.NextPageOffset == nil {
			break
		}
		nextOffset = result.Result.NextPageOffset
	}

	var ids []int
	for id := range uniqueIDs {
		ids = append(ids, id)
	}

	return ids, nil
}
