package llm

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

const (
	llamaPrefillCacheManifestVersion = 1

	// llamaPrefillCacheEraseTimeout bounds the slot erase that follows a failed
	// restore. The erase is best effort cleanup on a runner that is about to
	// serve requests, so it is kept short: leaving the slot dirty is recorded
	// as a warning, but blocking the load behind a wedged server is not.
	llamaPrefillCacheEraseTimeout = 10 * time.Second
)

type llamaPrefillCacheManifest struct {
	Version int   `json:"version"`
	Slots   []int `json:"slots"`
	// Checksums holds the SHA-256 of each slot file, parallel to Slots.
	Checksums []string `json:"checksums"`
}

type contextReader struct {
	ctx context.Context //nolint:containedctx // The reader checks cancellation between hash reads.
	r   io.Reader
}

var errInvalidLlamaPrefillCache = errors.New("invalid llama prefill cache")

func (r contextReader) Read(p []byte) (int, error) {
	if err := r.ctx.Err(); err != nil {
		return 0, err
	}
	return r.r.Read(p)
}

func fileSHA256(ctx context.Context, path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, contextReader{ctx: ctx, r: f}); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func (s *llamaServerRunner) SavePrefillCache(ctx context.Context) error {
	cachePath := s.launch.config.PrefillCachePath
	if cachePath == "" {
		return nil
	}
	if err := os.MkdirAll(cachePath, 0o700); err != nil {
		return fmt.Errorf("create prefill cache directory: %w", err)
	}
	manifestPath := filepath.Join(cachePath, "manifest.json")
	if err := os.Remove(manifestPath); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("invalidate old prefill cache manifest: %w", err)
	}

	if err := s.sem.Acquire(ctx, int64(s.launch.numParallel)); err != nil {
		return err
	}
	defer s.sem.Release(int64(s.launch.numParallel))

	start := time.Now()
	var savedBytes int64
	savedSlots := make([]int, 0, s.launch.numParallel)
	savedChecksums := make([]string, 0, s.launch.numParallel)
	for id := range s.launch.numParallel {
		finalName := fmt.Sprintf("slot-%d.bin", id)
		tempName := finalName + ".tmp"
		finalPath := filepath.Join(cachePath, finalName)
		tempPath := filepath.Join(cachePath, tempName)
		_ = os.Remove(tempPath)
		result, err := s.slotCacheAction(ctx, id, "save", tempName)
		if err != nil {
			var httpErr *llamaSlotCacheHTTPError
			if errors.As(err, &httpErr) && httpErr.StatusCode == http.StatusNotImplemented {
				_ = os.Remove(tempPath)
				if err := os.Remove(finalPath); err != nil && !os.IsNotExist(err) {
					return fmt.Errorf("remove unsupported slot %d prefill cache: %w", id, err)
				}
				continue
			}
			return err
		}
		if result.NSaved == 0 {
			_ = os.Remove(tempPath)
			if err := os.Remove(finalPath); err != nil && !os.IsNotExist(err) {
				return fmt.Errorf("remove empty slot %d prefill cache: %w", id, err)
			}
			continue
		}
		if err := os.Remove(finalPath); err != nil && !os.IsNotExist(err) {
			return fmt.Errorf("replace slot %d prefill cache: %w", id, err)
		}
		if err := os.Rename(tempPath, finalPath); err != nil {
			return fmt.Errorf("commit slot %d prefill cache: %w", id, err)
		}
		if info, err := os.Stat(finalPath); err == nil {
			savedBytes += info.Size()
		}
		sum, err := fileSHA256(ctx, finalPath)
		if err != nil {
			return fmt.Errorf("checksum slot %d prefill cache: %w", id, err)
		}
		savedSlots = append(savedSlots, id)
		savedChecksums = append(savedChecksums, sum)
	}

	manifest, err := json.Marshal(llamaPrefillCacheManifest{
		Version:   llamaPrefillCacheManifestVersion,
		Slots:     savedSlots,
		Checksums: savedChecksums,
	})
	if err != nil {
		return err
	}
	tempManifest := filepath.Join(cachePath, "manifest.json.tmp")
	if err := os.WriteFile(tempManifest, manifest, 0o600); err != nil {
		return fmt.Errorf("write prefill cache manifest: %w", err)
	}
	if err := os.Rename(tempManifest, manifestPath); err != nil {
		return fmt.Errorf("commit prefill cache manifest: %w", err)
	}
	if len(savedSlots) > 0 {
		slog.Info("saved prefill cache", "slots", len(savedSlots), "bytes", savedBytes, "duration", time.Since(start))
	}
	return nil
}

func (s *llamaServerRunner) RestorePrefillCache(ctx context.Context) error {
	err := s.restorePrefillCache(ctx)
	if errors.Is(err, errInvalidLlamaPrefillCache) && s.launch.config.PrefillCachePath != "" {
		_ = os.RemoveAll(s.launch.config.PrefillCachePath)
	}
	return err
}

func (s *llamaServerRunner) restorePrefillCache(ctx context.Context) error {
	cachePath := s.launch.config.PrefillCachePath
	if cachePath == "" {
		return nil
	}

	data, err := os.ReadFile(filepath.Join(cachePath, "manifest.json"))
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("read prefill cache manifest: %w", err)
	}
	var manifest llamaPrefillCacheManifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		return fmt.Errorf("%w: decode manifest: %v", errInvalidLlamaPrefillCache, err)
	}
	if manifest.Version != llamaPrefillCacheManifestVersion {
		return fmt.Errorf("%w: incompatible manifest version %d", errInvalidLlamaPrefillCache, manifest.Version)
	}
	seen := make(map[int]bool, len(manifest.Slots))
	for _, id := range manifest.Slots {
		if id < 0 || id >= s.launch.numParallel || seen[id] {
			return fmt.Errorf("%w: incompatible slot %d", errInvalidLlamaPrefillCache, id)
		}
		seen[id] = true
	}
	if len(manifest.Checksums) != len(manifest.Slots) {
		return fmt.Errorf("%w: incompatible manifest checksums", errInvalidLlamaPrefillCache)
	}
	// Verify every slot before the server reads any of them.
	for i, id := range manifest.Slots {
		sum, err := fileSHA256(ctx, filepath.Join(cachePath, fmt.Sprintf("slot-%d.bin", id)))
		if err != nil {
			if errors.Is(err, os.ErrNotExist) {
				return fmt.Errorf("%w: slot %d is missing", errInvalidLlamaPrefillCache, id)
			}
			return fmt.Errorf("read slot %d prefill cache: %w", id, err)
		}
		if sum != manifest.Checksums[i] {
			return fmt.Errorf("%w: checksum mismatch for slot %d", errInvalidLlamaPrefillCache, id)
		}
	}

	if err := s.sem.Acquire(ctx, int64(s.launch.numParallel)); err != nil {
		return err
	}
	defer s.sem.Release(int64(s.launch.numParallel))

	start := time.Now()
	for _, id := range manifest.Slots {
		if _, err := s.slotCacheAction(ctx, id, "restore", fmt.Sprintf("slot-%d.bin", id)); err != nil {
			// Clear any partial state before falling back to cold prefill.
			s.eraseSlotAfterFailedRestore(id)
			var httpErr *llamaSlotCacheHTTPError
			if errors.As(err, &httpErr) && httpErr.StatusCode == http.StatusBadRequest {
				return fmt.Errorf("%w: %v", errInvalidLlamaPrefillCache, err)
			}
			return err
		}
	}
	if len(manifest.Slots) > 0 {
		slog.Info("restored prefill cache", "slots", len(manifest.Slots), "duration", time.Since(start))
	}
	return nil
}

// eraseSlotAfterFailedRestore uses its own context: the restore may have
// failed precisely because the caller's context was canceled, and the erase
// must still run to clear any partially restored state.
func (s *llamaServerRunner) eraseSlotAfterFailedRestore(id int) {
	ctx, cancel := context.WithTimeout(context.Background(), llamaPrefillCacheEraseTimeout)
	defer cancel()
	if _, err := s.slotCacheAction(ctx, id, "erase", ""); err != nil {
		slog.Warn("failed to erase slot after failed prefill cache restore; slot may hold partial state", "slot", id, "error", err)
	}
}

type llamaSlotCacheHTTPError struct {
	StatusCode int
	Action     string
	Slot       int
	Message    string
}

func (e *llamaSlotCacheHTTPError) Error() string {
	return fmt.Sprintf("%s slot %d prefill cache: status %d: %s", e.Action, e.Slot, e.StatusCode, e.Message)
}

type llamaSlotCacheActionResponse struct {
	NSaved int `json:"n_saved"`
}

func (s *llamaServerRunner) slotCacheAction(ctx context.Context, id int, action, filename string) (llamaSlotCacheActionResponse, error) {
	body, err := json.Marshal(map[string]string{"filename": filename})
	if err != nil {
		return llamaSlotCacheActionResponse{}, err
	}
	url := fmt.Sprintf("http://127.0.0.1:%d/slots/%d?action=%s", s.port, id, action)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return llamaSlotCacheActionResponse{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := s.httpClient().Do(req)
	if err != nil {
		return llamaSlotCacheActionResponse{}, fmt.Errorf("%s slot %d prefill cache: %w", action, id, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		message, _ := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
		return llamaSlotCacheActionResponse{}, &llamaSlotCacheHTTPError{
			StatusCode: resp.StatusCode,
			Action:     action,
			Slot:       id,
			Message:    string(bytes.TrimSpace(message)),
		}
	}
	var result llamaSlotCacheActionResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return llamaSlotCacheActionResponse{}, fmt.Errorf("decode %s slot %d prefill cache response: %w", action, id, err)
	}
	return result, nil
}
