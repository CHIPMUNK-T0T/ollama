package mlxrunner

import (
	"context"
	"fmt"
	"io"
	"net/http"
)

func (c *Client) SavePrefillCache(ctx context.Context) error {
	return c.prefillCacheAction(ctx, "save")
}

func (c *Client) RestorePrefillCache(ctx context.Context) error {
	return c.prefillCacheAction(ctx, "restore")
}

func (c *Client) prefillCacheAction(ctx context.Context, action string) error {
	if c.prefillCachePath == "" {
		return nil
	}
	url := fmt.Sprintf("http://127.0.0.1:%d/internal/prefill-cache/%s", c.port, action)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, nil)
	if err != nil {
		return err
	}
	resp, err := c.client.Do(req)
	if err != nil {
		return fmt.Errorf("MLX prefill cache %s: %w", action, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
		return fmt.Errorf("MLX prefill cache %s: status %d: %s", action, resp.StatusCode, body)
	}
	return nil
}
