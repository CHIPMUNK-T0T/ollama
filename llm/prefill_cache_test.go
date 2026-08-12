package llm

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"golang.org/x/sync/semaphore"
)

func mustExist(t *testing.T, path string) {
	t.Helper()
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("stat %s: %v", path, err)
	}
}

func mustNotExist(t *testing.T, path string) {
	t.Helper()
	if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("stat %s: got %v, want %v", path, err, os.ErrNotExist)
	}
}

func mustWriteFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

// serverPort returns the port httptest bound, which the runner needs because
// it addresses llama-server by port rather than by URL.
func serverPort(t *testing.T, server *httptest.Server) int {
	t.Helper()
	parsed, err := url.Parse(server.URL)
	if err != nil {
		t.Fatalf("parse %s: %v", server.URL, err)
	}
	_, portText, err := net.SplitHostPort(parsed.Host)
	if err != nil {
		t.Fatalf("split %s: %v", parsed.Host, err)
	}
	port, err := strconv.Atoi(portText)
	if err != nil {
		t.Fatalf("parse port %s: %v", portText, err)
	}
	return port
}

func TestLlamaPrefillCacheSaveRestore(t *testing.T) {
	cachePath := t.TempDir()
	var mu sync.Mutex
	var actions []string

	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Filename string `json:"filename"`
		}
		// t.Fatalf must not run outside the test goroutine, so handler
		// failures are reported and answered with an error status.
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode slot request: %v", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		action := r.URL.Query().Get("action")
		id := strings.TrimPrefix(r.URL.Path, "/slots/")
		mu.Lock()
		actions = append(actions, action+":"+id+":"+body.Filename)
		mu.Unlock()
		if action == "save" {
			if err := os.WriteFile(filepath.Join(cachePath, body.Filename), []byte("slot "+id), 0o600); err != nil {
				t.Errorf("write %s: %v", body.Filename, err)
				w.WriteHeader(http.StatusInternalServerError)
				return
			}
		}
		w.Header().Set("Content-Type", "application/json")
		if action == "save" && id == "1" {
			w.WriteHeader(http.StatusNotImplemented)
			_, _ = w.Write([]byte(`{"error":"media slot"}`))
			return
		}
		if id == "0" {
			_, _ = w.Write([]byte(`{"n_saved":3}`))
		} else {
			_, _ = w.Write([]byte(`{"n_saved":0}`))
		}
	})
	server := httptest.NewServer(handler)
	defer server.Close()

	runner := &llamaServerRunner{
		port:   serverPort(t, server),
		client: server.Client(),
		sem:    semaphore.NewWeighted(2),
		launch: llamaServerLaunchConfig{
			numParallel: 2,
			config:      LlamaServerConfig{PrefillCachePath: cachePath},
		},
	}

	// An unsupported slot must not retain its previous snapshot.
	mustWriteFile(t, filepath.Join(cachePath, "slot-1.bin"), "stale")

	if err := runner.SavePrefillCache(t.Context()); err != nil {
		t.Fatalf("SavePrefillCache() = %v, want nil", err)
	}
	mustExist(t, filepath.Join(cachePath, "slot-0.bin"))
	mustNotExist(t, filepath.Join(cachePath, "slot-1.bin"))
	mustExist(t, filepath.Join(cachePath, "manifest.json"))
	mustNotExist(t, filepath.Join(cachePath, "slot-0.bin.tmp"))
	mustNotExist(t, filepath.Join(cachePath, "slot-1.bin.tmp"))

	// Replacing an existing published checkpoint must also work on Windows.
	if err := runner.SavePrefillCache(t.Context()); err != nil {
		t.Fatalf("second SavePrefillCache() = %v, want nil", err)
	}
	mustExist(t, filepath.Join(cachePath, "manifest.json"))

	mu.Lock()
	actions = nil
	mu.Unlock()
	if err := runner.RestorePrefillCache(t.Context()); err != nil {
		t.Fatalf("RestorePrefillCache() = %v, want nil", err)
	}
	mu.Lock()
	defer mu.Unlock()
	if want := []string{"restore:0:slot-0.bin"}; !slices.Equal(actions, want) {
		t.Fatalf("slot actions = %v, want %v", actions, want)
	}
}

func TestLlamaPrefillCacheSkipsEmptySlot(t *testing.T) {
	cachePath := t.TempDir()
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Filename string `json:"filename"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode slot request: %v", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if err := os.WriteFile(filepath.Join(cachePath, body.Filename), []byte("empty slot"), 0o600); err != nil {
			t.Errorf("write %s: %v", body.Filename, err)
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"n_saved":0}`))
	})
	server := httptest.NewServer(handler)
	defer server.Close()

	runner := &llamaServerRunner{
		port:   serverPort(t, server),
		client: server.Client(),
		sem:    semaphore.NewWeighted(1),
		launch: llamaServerLaunchConfig{
			numParallel: 1,
			config:      LlamaServerConfig{PrefillCachePath: cachePath},
		},
	}

	mustWriteFile(t, filepath.Join(cachePath, "slot-0.bin"), "stale")
	if err := runner.SavePrefillCache(t.Context()); err != nil {
		t.Fatalf("SavePrefillCache() = %v, want nil", err)
	}
	mustNotExist(t, filepath.Join(cachePath, "slot-0.bin.tmp"))
	mustNotExist(t, filepath.Join(cachePath, "slot-0.bin"))

	data, err := os.ReadFile(filepath.Join(cachePath, "manifest.json"))
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	var manifest llamaPrefillCacheManifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		t.Fatalf("decode manifest: %v", err)
	}
	if len(manifest.Slots) != 0 || len(manifest.Checksums) != 0 {
		t.Fatalf("manifest = %+v, want no saved slots", manifest)
	}
}

func manifestForSlots(contents ...string) string {
	manifest := llamaPrefillCacheManifest{Version: llamaPrefillCacheManifestVersion}
	for id, content := range contents {
		sum := sha256.Sum256([]byte(content))
		manifest.Slots = append(manifest.Slots, id)
		manifest.Checksums = append(manifest.Checksums, hex.EncodeToString(sum[:]))
	}
	data, err := json.Marshal(manifest)
	if err != nil {
		panic(err)
	}
	return string(data)
}

func manifestForSlot0(content string) string {
	return manifestForSlots(content)
}

func TestFileSHA256HonorsCancellation(t *testing.T) {
	path := filepath.Join(t.TempDir(), "slot.bin")
	mustWriteFile(t, path, "kv")
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := fileSHA256(ctx, path); !errors.Is(err, context.Canceled) {
		t.Fatalf("fileSHA256() = %v, want %v", err, context.Canceled)
	}
}

func TestLlamaPrefillCacheErasesSlotOnFailedRestore(t *testing.T) {
	cachePath := t.TempDir()
	mustWriteFile(t, filepath.Join(cachePath, "manifest.json"), manifestForSlot0("kv"))
	mustWriteFile(t, filepath.Join(cachePath, "slot-0.bin"), "kv")

	var mu sync.Mutex
	var actions []string
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		action := r.URL.Query().Get("action")
		id := strings.TrimPrefix(r.URL.Path, "/slots/")
		mu.Lock()
		actions = append(actions, action+":"+id)
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		if action == "restore" {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"error":"invalid slot save file"}`))
			return
		}
		_, _ = w.Write([]byte(`{}`))
	})
	server := httptest.NewServer(handler)
	defer server.Close()

	runner := &llamaServerRunner{
		port:   serverPort(t, server),
		client: server.Client(),
		sem:    semaphore.NewWeighted(1),
		launch: llamaServerLaunchConfig{
			numParallel: 1,
			config:      LlamaServerConfig{PrefillCachePath: cachePath},
		},
	}

	if err := runner.RestorePrefillCache(t.Context()); err == nil {
		t.Fatal("RestorePrefillCache() = nil, want an error")
	}
	// The slot must be erased so a partial restore cannot corrupt later
	// completions, and the definitive failure invalidates the snapshot.
	mu.Lock()
	if want := []string{"restore:0", "erase:0"}; !slices.Equal(actions, want) {
		mu.Unlock()
		t.Fatalf("slot actions = %v, want %v", actions, want)
	}
	mu.Unlock()
	mustNotExist(t, cachePath)
}

func TestLlamaPrefillCacheChecksumMismatchFailsOpen(t *testing.T) {
	cachePath := t.TempDir()
	mustWriteFile(t, filepath.Join(cachePath, "manifest.json"), manifestForSlots("kv0", "kv1"))
	mustWriteFile(t, filepath.Join(cachePath, "slot-0.bin"), "kv0")
	mustWriteFile(t, filepath.Join(cachePath, "slot-1.bin"), "corrupted")

	var actions atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		actions.Add(1)
		_, _ = w.Write([]byte(`{}`))
	}))
	defer server.Close()
	runner := &llamaServerRunner{
		port:   serverPort(t, server),
		client: server.Client(),
		sem:    semaphore.NewWeighted(2),
		launch: llamaServerLaunchConfig{
			numParallel: 2,
			config:      LlamaServerConfig{PrefillCachePath: cachePath},
		},
	}
	err := runner.RestorePrefillCache(t.Context())
	if err == nil || !strings.Contains(err.Error(), "checksum mismatch") {
		t.Fatalf("RestorePrefillCache() = %v, want a checksum mismatch error", err)
	}
	if got := actions.Load(); got != 0 {
		t.Fatalf("slot actions = %d, want 0: every checksum must pass before restore", got)
	}
	mustNotExist(t, cachePath)
}

func TestLlamaPrefillCacheKeepsSnapshotOnTransientRestoreFailure(t *testing.T) {
	cachePath := t.TempDir()
	mustWriteFile(t, filepath.Join(cachePath, "manifest.json"), manifestForSlot0("kv"))
	mustWriteFile(t, filepath.Join(cachePath, "slot-0.bin"), "kv")

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("action") == "restore" {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		_, _ = w.Write([]byte(`{}`))
	}))
	defer server.Close()
	runner := &llamaServerRunner{
		port:   serverPort(t, server),
		client: server.Client(),
		sem:    semaphore.NewWeighted(1),
		launch: llamaServerLaunchConfig{
			numParallel: 1,
			config:      LlamaServerConfig{PrefillCachePath: cachePath},
		},
	}
	if err := runner.RestorePrefillCache(t.Context()); err == nil {
		t.Fatal("RestorePrefillCache() = nil, want an error")
	}
	mustExist(t, filepath.Join(cachePath, "manifest.json"))
	mustExist(t, filepath.Join(cachePath, "slot-0.bin"))
}

func TestLlamaPrefillCacheKeepsSnapshotOnInterruptedRestore(t *testing.T) {
	cachePath := t.TempDir()
	mustWriteFile(t, filepath.Join(cachePath, "manifest.json"), manifestForSlot0("kv"))
	mustWriteFile(t, filepath.Join(cachePath, "slot-0.bin"), "kv")

	runner := &llamaServerRunner{
		sem: semaphore.NewWeighted(1),
		launch: llamaServerLaunchConfig{
			numParallel: 1,
			config:      LlamaServerConfig{PrefillCachePath: cachePath},
		},
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if err := runner.RestorePrefillCache(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("RestorePrefillCache() = %v, want %v", err, context.Canceled)
	}
	mustExist(t, filepath.Join(cachePath, "manifest.json"))
	mustExist(t, filepath.Join(cachePath, "slot-0.bin"))
}

func TestLlamaPrefillCacheRejectsIncompatibleManifest(t *testing.T) {
	cachePath := t.TempDir()
	mustWriteFile(t, filepath.Join(cachePath, "manifest.json"), `{"version":1,"slots":[0,1]}`)

	runner := &llamaServerRunner{
		sem: semaphore.NewWeighted(1),
		launch: llamaServerLaunchConfig{
			numParallel: 1,
			config:      LlamaServerConfig{PrefillCachePath: cachePath},
		},
	}
	err := runner.RestorePrefillCache(t.Context())
	if err == nil || !strings.Contains(err.Error(), "incompatible") {
		t.Fatalf("RestorePrefillCache() = %v, want an incompatible manifest error", err)
	}
	mustNotExist(t, cachePath)
}

func TestAppendPrefillCacheArgsFailOpen(t *testing.T) {
	unchanged := []string{"--model", "m"}
	if got := appendPrefillCacheArgs(unchanged, ""); !slices.Equal(got, unchanged) {
		t.Fatalf("appendPrefillCacheArgs(%v, \"\") = %v, want %v", unchanged, got, unchanged)
	}

	cachePath := filepath.Join(t.TempDir(), "cache")
	want := []string{"--slot-save-path", filepath.Clean(cachePath) + string(os.PathSeparator)}
	if got := appendPrefillCacheArgs(nil, cachePath); !slices.Equal(got, want) {
		t.Fatalf("appendPrefillCacheArgs(nil, %q) = %v, want %v", cachePath, got, want)
	}
	mustExist(t, cachePath)

	// A path that cannot be created disables persistence instead of failing
	// the load.
	blocked := filepath.Join(t.TempDir(), "blocked")
	mustWriteFile(t, blocked, "")
	if got := appendPrefillCacheArgs(nil, filepath.Join(blocked, "cache")); len(got) != 0 {
		t.Fatalf("appendPrefillCacheArgs(nil, <uncreatable>) = %v, want no args", got)
	}
}
