package server

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/llm"
)

const (
	maxPrefillCacheDiskBytes   int64 = 8 << 30
	prefillCachePersistTimeout       = time.Minute
	prefillCacheRootPrefix           = "ollama-prefill-cache-"
)

func initPrefillCacheRoot(ctx context.Context) string {
	if !envconfig.PrefillCache() {
		return ""
	}

	root := prefillCacheRootForPid(os.Getpid())
	// A leftover directory for this pid predates the process and must not be restored.
	if err := os.RemoveAll(root); err != nil {
		slog.Warn("prefill cache persistence disabled", "error", err)
		return ""
	}
	if err := os.MkdirAll(root, 0o700); err != nil {
		slog.Warn("prefill cache persistence disabled", "error", err)
		return ""
	}

	go sweepStalePrefillCacheRoots(filepath.Dir(root), root)
	go func() {
		<-ctx.Done()
		if err := os.RemoveAll(root); err != nil {
			slog.Debug("failed to remove prefill cache directory", "path", root, "error", err)
		}
	}()

	return root
}

func prefillCacheRootForPid(pid int) string {
	return filepath.Join(os.TempDir(), prefillCacheRootPrefix+strconv.Itoa(pid))
}

// sweepStalePrefillCacheRoots removes cache roots left behind by daemons that
// exited without cleanup (crash, SIGKILL): those whose encoded pid is dead.
// Roots without a numeric pid suffix are left alone.
func sweepStalePrefillCacheRoots(base, keep string) {
	entries, err := os.ReadDir(base)
	if err != nil {
		return
	}
	for _, entry := range entries {
		name := entry.Name()
		if !entry.IsDir() || !strings.HasPrefix(name, prefillCacheRootPrefix) {
			continue
		}
		path := filepath.Join(base, name)
		if path == keep {
			continue
		}
		pid, err := strconv.Atoi(strings.TrimPrefix(name, prefillCacheRootPrefix))
		if err != nil || pidAlive(pid) {
			continue
		}
		if err := os.RemoveAll(path); err != nil {
			slog.Debug("failed to remove stale prefill cache", "path", path, "error", err)
		} else {
			slog.Info("removed stale prefill cache", "path", path)
		}
	}
}

func pidAlive(pid int) bool {
	if pid <= 0 {
		return false
	}
	proc, err := os.FindProcess(pid)
	if err != nil {
		// Windows: FindProcess opens a handle and fails if the process is gone.
		return false
	}
	if runtime.GOOS == "windows" {
		return true
	}
	// EPERM means the pid exists but belongs to another user; treat it as
	// alive so we never delete a foreign daemon's cache.
	err = proc.Signal(syscall.Signal(0))
	return err == nil || errors.Is(err, syscall.EPERM)
}

type prefillCacheIdentity struct {
	Version      int        `json:"version"`
	Runner       string     `json:"runner"`
	Model        string     `json:"model"`
	Draft        string     `json:"draft,omitempty"`
	Adapters     []string   `json:"adapters,omitempty"`
	Projectors   []string   `json:"projectors,omitempty"`
	Options      api.Runner `json:"options"`
	NumParallel  int        `json:"num_parallel"`
	KVCacheType  string     `json:"kv_cache_type,omitempty"`
	ContextShift bool       `json:"context_shift,omitempty"`
}

func (s *Scheduler) prefillCachePath(identity prefillCacheIdentity) string {
	if s.prefillCacheRoot == "" {
		return ""
	}
	data, err := json.Marshal(identity)
	if err != nil {
		slog.Debug("failed to identify prefill cache", "error", err)
		return ""
	}
	sum := sha256.Sum256(data)
	return filepath.Join(s.prefillCacheRoot, hex.EncodeToString(sum[:]))
}

func (s *Scheduler) llamaPrefillCachePath(req *LlmRequest, opts api.Runner, numParallel int) string {
	return s.prefillCachePath(prefillCacheIdentity{
		Version:      1,
		Runner:       "llama.cpp",
		Model:        schedulerModelKey(req.model),
		Draft:        req.model.DraftPath,
		Adapters:     req.model.AdapterPaths,
		Projectors:   req.model.ProjectorPaths,
		Options:      opts,
		NumParallel:  numParallel,
		KVCacheType:  envconfig.KvCacheType(),
		ContextShift: req.contextShift,
	})
}

func (s *Scheduler) mlxPrefillCachePath(req *LlmRequest) string {
	return s.prefillCachePath(prefillCacheIdentity{
		Version:     1,
		Runner:      "mlx",
		Model:       schedulerModelKey(req.model),
		Options:     req.opts.Runner,
		NumParallel: 1,
	})
}

func (runner *runnerRef) restorePrefillCache(ctx context.Context) {
	cache, ok := runner.llama.(llm.PrefillCachePersistor)
	if !ok {
		return
	}
	if runner.prefillCacheDir != "" {
		// Prefer other entries if another unload prunes while restore reads this one.
		now := time.Now()
		_ = os.Chtimes(runner.prefillCacheDir, now, now)
	}
	restoreCtx, cancel := context.WithTimeout(ctx, prefillCachePersistTimeout)
	defer cancel()
	if err := cache.RestorePrefillCache(restoreCtx); err != nil {
		slog.Warn(
			"failed to restore prefill cache; continuing with cold cache",
			"model", runner.modelKey,
			"error", err,
		)
	}
}

// savePrefillCache saves at most once. refMu must be held.
func (runner *runnerRef) savePrefillCache(ctx context.Context) {
	if runner.prefillCacheSaveAttempted || runner.llama == nil || runner.loading {
		return
	}
	runner.prefillCacheSaveAttempted = true
	if cache, ok := runner.llama.(llm.PrefillCachePersistor); ok && !runner.llama.HasExited() {
		saveCtx, cancel := context.WithTimeout(ctx, prefillCachePersistTimeout)
		defer cancel()
		if err := cache.SavePrefillCache(saveCtx); err != nil {
			slog.Warn("failed to save prefill cache; runner will unload without a snapshot", "model", runner.modelKey, "error", err)
		}
	}
}

// prunePrefillCache evicts the oldest entries except keep until under maxBytes.
func prunePrefillCache(root string, maxBytes int64, keep string) {
	if root == "" || maxBytes < 0 {
		return
	}
	type cacheDir struct {
		path    string
		size    int64
		modTime time.Time
	}
	entries, err := os.ReadDir(root)
	if err != nil {
		if !os.IsNotExist(err) {
			slog.Debug("failed to scan prefill cache", "path", root, "error", err)
		}
		return
	}
	var dirs []cacheDir
	var total int64
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		info, err := entry.Info()
		if err != nil {
			continue
		}
		dir := cacheDir{
			path:    filepath.Join(root, entry.Name()),
			size:    prefillCacheDiskUsage(filepath.Join(root, entry.Name())),
			modTime: info.ModTime(),
		}
		dirs = append(dirs, dir)
		total += dir.size
	}
	if total <= maxBytes {
		return
	}
	sort.Slice(dirs, func(i, j int) bool { return dirs[i].modTime.Before(dirs[j].modTime) })
	for _, dir := range dirs {
		if total <= maxBytes {
			break
		}
		if dir.path == keep {
			continue
		}
		if err := os.RemoveAll(dir.path); err != nil {
			// RemoveAll may have removed part of the directory before failing.
			total = prefillCacheDiskUsage(root)
			slog.Debug("failed to evict prefill cache", "path", dir.path, "error", err)
			continue
		}
		total -= dir.size
		slog.Debug("evicted prefill cache", "path", dir.path, "size", dir.size)
	}
}

func prefillCacheDiskUsage(path string) int64 {
	var size int64
	_ = filepath.Walk(path, func(_ string, info os.FileInfo, err error) error {
		if err == nil && info.Mode().IsRegular() {
			size += info.Size()
		}
		return nil
	})
	return size
}
