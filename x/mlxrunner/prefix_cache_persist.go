package mlxrunner

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"time"

	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

const (
	prefixCacheFileVersion      = 1
	prefixCacheMetadataKey      = "ollama.prefill_cache"
	prefixCacheFilename         = "active-prefix.safetensors"
	prefixCacheChecksumFilename = "active-prefix.checksum.json"
)

type prefixCacheFileManifest struct {
	Version        int                     `json:"version"`
	DraftLookahead int                     `json:"draft_lookahead"`
	Offset         int                     `json:"offset"`
	Tokens         []int64                 `json:"tokens"`
	Layers         []cache.PersistentState `json:"layers"`
}

// prefixCacheChecksum records the SHA-256 of the snapshot it sits next to.
// safetensors validates its own structure, so a truncated or header-damaged
// snapshot already fails to load. Corruption confined to the tensor payload
// passes every structural check, loads cleanly, and would silently poison every
// completion that reuses the restored prefix; only a checksum catches that.
type prefixCacheChecksum struct {
	Version int    `json:"version"`
	SHA256  string `json:"sha256"`
}

type contextReader struct {
	ctx context.Context //nolint:containedctx // The reader checks cancellation between hash reads.
	r   io.Reader
}

var errInvalidMLXPrefillCache = errors.New("invalid MLX prefill cache")

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

func (c *prefixCache) save(ctx context.Context, path string) error {
	if path == "" {
		return nil
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	start := time.Now()

	// Prefer the deepest captured snapshot on the active path. The live cache
	// normally includes generated tokens beyond the prompt. Position-sliceable
	// KV caches can rewind that state on the next request, but recurrent caches
	// cannot. Prefill already captures a restore point near the end of the
	// prompt, so make that state live before serializing it. This also keeps the
	// persisted token prefix aligned with every cache layer's exact state.
	for i := len(c.activePath) - 1; i > 0; i-- {
		node := c.activePath[i]
		if node.endOffset > 0 && hasAllSnapshots(node, c.caches) {
			c.switchToPath(c.activePath[:i+1], node.endOffset)
			break
		}
	}

	offset := c.minCacheOffset()
	if offset <= 0 || len(c.activePath) == 0 {
		if c.keepPersistedSnapshot {
			return nil
		}
		// Nothing to snapshot: drop any previous snapshot so the next load
		// does not restore a prefix state this cache no longer holds. Remove the
		// checksum first, so an interrupted invalidation leaves an unverifiable
		// snapshot (a cold miss) rather than a verifiable stale one.
		if err := os.Remove(filepath.Join(path, prefixCacheChecksumFilename)); err != nil && !os.IsNotExist(err) {
			return fmt.Errorf("invalidate stale MLX prefill cache checksum: %w", err)
		}
		if err := os.Remove(filepath.Join(path, prefixCacheFilename)); err != nil && !os.IsNotExist(err) {
			return fmt.Errorf("invalidate stale MLX prefill cache: %w", err)
		}
		return nil
	}

	tokens := make([]int64, 0, offset)
	for _, node := range c.activePath {
		for _, token := range node.tokens {
			tokens = append(tokens, int64(token))
		}
	}
	if len(tokens) < offset {
		return fmt.Errorf("active prefix has %d keys for cache offset %d", len(tokens), offset)
	}
	tokens = tokens[:offset]

	manifest := prefixCacheFileManifest{
		Version:        prefixCacheFileVersion,
		DraftLookahead: c.draftLookahead,
		Offset:         offset,
		Tokens:         tokens,
		Layers:         make([]cache.PersistentState, len(c.caches)),
	}
	arrays := make(map[string]*mlx.Array, 2*len(c.caches))
	toEval := make([]*mlx.Array, 0, 2*len(c.caches))
	for i, layer := range c.caches {
		if err := ctx.Err(); err != nil {
			return err
		}
		state, err := cache.ExportPersistentState(layer)
		if err != nil {
			return fmt.Errorf("export cache layer %d: %w", i, err)
		}
		manifest.Layers[i] = state
		for j, array := range state.Arrays {
			if array == nil {
				return fmt.Errorf("cache layer %d has nil state tensor %d", i, j)
			}
			arrays[fmt.Sprintf("layer.%d.%d", i, j)] = array
			toEval = append(toEval, array)
		}
	}
	metadata, err := json.Marshal(manifest)
	if err != nil {
		return err
	}

	if err := os.MkdirAll(path, 0o700); err != nil {
		return fmt.Errorf("create MLX prefill cache directory: %w", err)
	}
	mlx.Eval(toEval...)
	if err := ctx.Err(); err != nil {
		return err
	}
	// MLX appends .safetensors unless the supplied path already ends in that
	// suffix. Keep the temporary path suffixed so Rename sees the file MLX
	// actually created.
	temp := filepath.Join(path, prefixCacheFilename+".tmp.safetensors")
	_ = os.Remove(temp)
	defer func() { _ = os.Remove(temp) }()
	if err := mlx.SaveSafetensorsWithMetadata(temp, arrays, map[string]string{prefixCacheMetadataKey: string(metadata)}); err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	final := filepath.Join(path, prefixCacheFilename)
	// Rename replaces an existing file atomically. Do not remove the old
	// snapshot first: if the runner dies at any point before this rename, the
	// scheduler can still restore the last successfully committed snapshot.
	if err := os.Rename(temp, final); err != nil {
		return fmt.Errorf("commit MLX prefill cache: %w", err)
	}
	var savedBytes int64
	if info, err := os.Stat(final); err == nil {
		savedBytes = info.Size()
	}

	// Commit the checksum after the snapshot, not before. llama-server has to
	// invalidate its manifest first because it commits several slot files with
	// no single atomic point; this path commits exactly one file, so writing the
	// checksum last keeps the previous snapshot and its matching checksum usable
	// if the runner dies before the rename above. Dying between the two commits
	// leaves a snapshot the checksum no longer matches, which is a cold miss.
	if err := writePrefixCacheChecksum(ctx, path, final); err != nil {
		return err
	}
	c.keepPersistedSnapshot = false

	slog.Info("saved MLX prefill cache", "tokens", offset, "bytes", savedBytes, "duration", time.Since(start))
	return nil
}

func writePrefixCacheChecksum(ctx context.Context, path, snapshot string) error {
	sum, err := fileSHA256(ctx, snapshot)
	if err != nil {
		return fmt.Errorf("checksum MLX prefill cache: %w", err)
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	body, err := json.Marshal(prefixCacheChecksum{Version: prefixCacheFileVersion, SHA256: sum})
	if err != nil {
		return err
	}
	temp := filepath.Join(path, prefixCacheChecksumFilename+".tmp")
	defer func() { _ = os.Remove(temp) }()
	if err := os.WriteFile(temp, body, 0o600); err != nil {
		return fmt.Errorf("write MLX prefill cache checksum: %w", err)
	}
	if err := os.Rename(temp, filepath.Join(path, prefixCacheChecksumFilename)); err != nil {
		return fmt.Errorf("commit MLX prefill cache checksum: %w", err)
	}
	return nil
}

func (c *prefixCache) restore(ctx context.Context, path string) error {
	err := c.restoreFile(ctx, path)
	if err == nil {
		c.keepPersistedSnapshot = false
		return nil
	}
	if errors.Is(err, errInvalidMLXPrefillCache) {
		c.keepPersistedSnapshot = false
		if path != "" {
			_ = os.Remove(filepath.Join(path, prefixCacheFilename))
			_ = os.Remove(filepath.Join(path, prefixCacheChecksumFilename))
		}
	} else {
		c.keepPersistedSnapshot = true
	}
	return err
}

// verifyPrefixCacheChecksum reports whether snapshot still matches the SHA-256
// recorded beside it. A missing checksum is not a corruption signal: it means
// the save was interrupted between committing the snapshot and committing the
// checksum, so the snapshot is simply unusable and the caller runs cold.
func verifyPrefixCacheChecksum(ctx context.Context, path, snapshot string) (bool, error) {
	body, err := os.ReadFile(filepath.Join(path, prefixCacheChecksumFilename))
	if os.IsNotExist(err) {
		return false, nil
	}
	if err != nil {
		return false, fmt.Errorf("read MLX prefill cache checksum: %w", err)
	}
	var recorded prefixCacheChecksum
	if err := json.Unmarshal(body, &recorded); err != nil {
		return false, fmt.Errorf("%w: decode checksum: %v", errInvalidMLXPrefillCache, err)
	}
	if recorded.Version != prefixCacheFileVersion {
		return false, fmt.Errorf("%w: incompatible checksum version %d", errInvalidMLXPrefillCache, recorded.Version)
	}
	sum, err := fileSHA256(ctx, snapshot)
	if err != nil {
		return false, fmt.Errorf("checksum MLX prefill cache: %w", err)
	}
	if sum != recorded.SHA256 {
		return false, fmt.Errorf("%w: checksum mismatch", errInvalidMLXPrefillCache)
	}
	return true, nil
}

func (c *prefixCache) restoreFile(ctx context.Context, path string) error {
	if path == "" {
		return nil
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if c.root != nil {
		// Restore only targets a freshly loaded runner; replacing a live trie
		// would leak its pinned arrays and desync pagedOutBytes.
		return fmt.Errorf("refusing to restore MLX prefill cache over a warm cache")
	}
	start := time.Now()
	filename := filepath.Join(path, prefixCacheFilename)
	if _, err := os.Stat(filename); os.IsNotExist(err) {
		return nil
	} else if err != nil {
		return fmt.Errorf("stat MLX prefill cache: %w", err)
	}

	// Verify before MLX reads the file. Nothing is allocated on the device for a
	// snapshot that fails here, and the rejection is exercisable on hosts
	// without an MLX runtime.
	verified, err := verifyPrefixCacheChecksum(ctx, path, filename)
	if err != nil {
		return err
	}
	if !verified {
		slog.Warn("MLX prefill cache snapshot has no checksum; ignoring it", "path", filename)
		return nil
	}

	file, err := mlx.LoadSafetensorsNative(filename)
	if err != nil {
		return fmt.Errorf("%w: load snapshot: %v", errInvalidMLXPrefillCache, err)
	}
	defer file.Free()
	if err := ctx.Err(); err != nil {
		return err
	}

	var manifest prefixCacheFileManifest
	if err := json.Unmarshal([]byte(file.GetMetadata(prefixCacheMetadataKey)), &manifest); err != nil {
		return fmt.Errorf("%w: decode metadata: %v", errInvalidMLXPrefillCache, err)
	}
	if manifest.Version != prefixCacheFileVersion || manifest.DraftLookahead != c.draftLookahead {
		return fmt.Errorf("%w: incompatible version or draft lookahead", errInvalidMLXPrefillCache)
	}
	if manifest.Offset <= 0 || len(manifest.Tokens) != manifest.Offset || len(manifest.Layers) != len(c.caches) {
		return fmt.Errorf("%w: incompatible dimensions", errInvalidMLXPrefillCache)
	}

	for i := range manifest.Layers {
		if err := ctx.Err(); err != nil {
			c.freeAll()
			return err
		}
		state := manifest.Layers[i]
		if state.Kind != "nil" {
			state.Arrays = []*mlx.Array{
				file.Get(fmt.Sprintf("layer.%d.0", i)),
				file.Get(fmt.Sprintf("layer.%d.1", i)),
			}
		}
		if err := cache.ImportPersistentState(c.caches[i], state); err != nil {
			c.freeAll()
			return fmt.Errorf("%w: restore cache layer %d: %v", errInvalidMLXPrefillCache, i, err)
		}
	}
	if err := ctx.Err(); err != nil {
		c.freeAll()
		return err
	}

	now := time.Now()
	root := &trieNode{lastUsed: now}
	tokens := make([]trieKey, len(manifest.Tokens))
	for i, token := range manifest.Tokens {
		tokens[i] = trieKey(token)
	}
	leaf := &trieNode{
		tokens:    tokens,
		endOffset: manifest.Offset,
		parent:    root,
		lastUsed:  now,
		user:      true,
	}
	root.children = []*trieNode{leaf}
	c.root = root
	c.activePath = []*trieNode{root, leaf}
	c.pagedOutBytes = 0

	// Keep the restored boundary reusable after this runner generates new
	// tokens. Without a trie snapshot, advancePath extends the restored leaf
	// in place and a later save can only see the post-generation recurrent
	// state, which cannot be rewound to the prompt boundary. Marking the leaf
	// as a user restore point also prevents that in-place extension.
	snapshots := make([]cache.Snapshot, len(c.caches))
	for i, layer := range c.caches {
		if layer != nil {
			snapshots[i] = layer.Snapshot(0)
		}
	}
	leaf.setSnapshots(snapshots, &c.pagedOutBytes)

	slog.Info("restored MLX prefill cache", "tokens", manifest.Offset, "duration", time.Since(start))
	return nil
}
