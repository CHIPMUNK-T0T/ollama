package server

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
)

func mustMkdir(t *testing.T, path string) {
	t.Helper()
	if err := os.Mkdir(path, 0o700); err != nil {
		t.Fatalf("mkdir %s: %v", path, err)
	}
}

func mustWriteFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

func mustDirExist(t *testing.T, path string) {
	t.Helper()
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat %s: %v", path, err)
	}
	if !info.IsDir() {
		t.Fatalf("%s is not a directory", path)
	}
}

func mustNotExist(t *testing.T, path string) {
	t.Helper()
	if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("stat %s: got %v, want %v", path, err, os.ErrNotExist)
	}
}

type prefillCacheMock struct {
	*mockLlm
	calls        []string
	restoreErr   error
	saveErr      error
	saveStarted  chan struct{}
	saveContinue chan struct{}
}

func (m *prefillCacheMock) RestorePrefillCache(context.Context) error {
	m.calls = append(m.calls, "restore")
	return m.restoreErr
}

func (m *prefillCacheMock) SavePrefillCache(ctx context.Context) error {
	m.calls = append(m.calls, "save")
	if m.saveStarted != nil {
		close(m.saveStarted)
		select {
		case <-m.saveContinue:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	return m.saveErr
}

func (m *prefillCacheMock) Close() error {
	m.calls = append(m.calls, "close")
	return m.mockLlm.Close()
}

func TestInitSchedulerPrefillCacheGate(t *testing.T) {
	temp := t.TempDir()
	t.Setenv("TMPDIR", temp)
	t.Setenv("TMP", temp)
	t.Setenv("TEMP", temp)

	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()

	s := InitScheduler(ctx)
	if s.prefillCacheRoot != "" {
		t.Fatalf("prefillCacheRoot = %q, want empty: persistence must be opt-in", s.prefillCacheRoot)
	}

	t.Setenv("OLLAMA_PREFILL_CACHE", "1")
	s = InitScheduler(ctx)
	if want := prefillCacheRootForPid(os.Getpid()); s.prefillCacheRoot != want {
		t.Fatalf("prefillCacheRoot = %q, want %q", s.prefillCacheRoot, want)
	}
	mustDirExist(t, s.prefillCacheRoot)

	// A pre-existing directory for this pid predates the process and must be
	// emptied, not reused.
	stale := filepath.Join(s.prefillCacheRoot, "stale")
	mustMkdir(t, stale)
	s = InitScheduler(ctx)
	mustDirExist(t, s.prefillCacheRoot)
	mustNotExist(t, stale)

	if err := os.RemoveAll(s.prefillCacheRoot); err != nil {
		t.Fatalf("remove %s: %v", s.prefillCacheRoot, err)
	}
}

func TestSweepStalePrefillCacheRoots(t *testing.T) {
	base := t.TempDir()
	own := filepath.Join(base, prefillCacheRootPrefix+strconv.Itoa(os.Getpid()))
	dead := filepath.Join(base, prefillCacheRootPrefix+"2147483647")
	legacy := filepath.Join(base, prefillCacheRootPrefix+"not-a-pid")
	unrelated := filepath.Join(base, "unrelated")
	for _, dir := range []string{own, dead, legacy, unrelated} {
		mustMkdir(t, dir)
	}

	sweepStalePrefillCacheRoots(base, own)
	mustDirExist(t, own)
	mustNotExist(t, dead)
	mustDirExist(t, legacy)
	mustDirExist(t, unrelated)
}

func TestPrefillCachePathIdentity(t *testing.T) {
	s := &Scheduler{prefillCacheRoot: t.TempDir()}
	identity := prefillCacheIdentity{
		Version:     1,
		Runner:      "llama.cpp",
		Model:       "sha256-abc",
		Options:     api.Runner{NumCtx: 8192},
		NumParallel: 1,
	}
	path := s.prefillCachePath(identity)
	if path == "" {
		t.Fatal("prefillCachePath() = empty, want a path")
	}
	if got := s.prefillCachePath(identity); got != path {
		t.Fatalf("prefillCachePath() = %q, want stable path %q", got, path)
	}
	identity.Model = "sha256-def"
	if got := s.prefillCachePath(identity); got == path {
		t.Fatal("prefillCachePath() did not change with model identity")
	}
}

func TestPrunePrefillCache(t *testing.T) {
	root := t.TempDir()
	oldDir := filepath.Join(root, "old")
	newDir := filepath.Join(root, "new")
	mustMkdir(t, oldDir)
	mustMkdir(t, newDir)
	mustWriteFile(t, filepath.Join(oldDir, "cache"), "1234")
	mustWriteFile(t, filepath.Join(newDir, "cache"), "5678")
	now := time.Now()
	for dir, mtime := range map[string]time.Time{oldDir: now.Add(-time.Hour), newDir: now} {
		if err := os.Chtimes(dir, mtime, mtime); err != nil {
			t.Fatalf("chtimes %s: %v", dir, err)
		}
	}

	prunePrefillCache(root, 4, "")
	mustNotExist(t, oldDir)
	mustDirExist(t, newDir)
}

func TestPrunePrefillCacheExemptsCurrentModel(t *testing.T) {
	root := t.TempDir()
	current := filepath.Join(root, "current")
	mustMkdir(t, current)
	mustWriteFile(t, filepath.Join(current, "cache"), "12345678")

	// A snapshot larger than the cap must not evict itself right after save.
	prunePrefillCache(root, 4, current)
	mustDirExist(t, current)

	// Other models' caches are still evicted, oldest first.
	oldDir := filepath.Join(root, "old")
	mustMkdir(t, oldDir)
	mustWriteFile(t, filepath.Join(oldDir, "cache"), "1234")
	past := time.Now().Add(-time.Hour)
	if err := os.Chtimes(oldDir, past, past); err != nil {
		t.Fatalf("chtimes %s: %v", oldDir, err)
	}
	prunePrefillCache(root, 4, current)
	mustNotExist(t, oldDir)
	mustDirExist(t, current)
}

func TestRunnerSavePrefillCacheRunsOnce(t *testing.T) {
	persist := &prefillCacheMock{mockLlm: &mockLlm{}}
	runner := &runnerRef{llama: persist}
	runner.refMu.Lock()
	defer runner.refMu.Unlock()

	// The expired path saves before waiting for VRAM recovery; the unload
	// that follows must not save a second time.
	runner.savePrefillCache(t.Context())
	runner.unload()
	if want := []string{"save", "close"}; !slices.Equal(persist.calls, want) {
		t.Fatalf("runner calls = %v, want %v", persist.calls, want)
	}
}

func TestRunnerSavePrefillCacheFailureRunsOnce(t *testing.T) {
	persist := &prefillCacheMock{mockLlm: &mockLlm{}, saveErr: errors.New("disk full")}
	runner := &runnerRef{llama: persist}
	runner.refMu.Lock()
	defer runner.refMu.Unlock()

	runner.savePrefillCache(t.Context())
	runner.unload()
	if want := []string{"save", "close"}; !slices.Equal(persist.calls, want) {
		t.Fatalf("runner calls = %v, want %v", persist.calls, want)
	}
}

func TestRunnerSavePrefillCacheSkipsLoadingRunner(t *testing.T) {
	persist := &prefillCacheMock{mockLlm: &mockLlm{}}
	runner := &runnerRef{llama: persist, loading: true}
	runner.refMu.Lock()
	defer runner.refMu.Unlock()

	runner.unload()
	if want := []string{"close"}; !slices.Equal(persist.calls, want) {
		t.Fatalf("runner calls = %v, want %v", persist.calls, want)
	}
}

func TestSchedSavesPrefillCacheBeforeVRAMRecoveryWithoutLoadedLock(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()

	persist := &prefillCacheMock{
		mockLlm:      &mockLlm{},
		saveStarted:  make(chan struct{}),
		saveContinue: make(chan struct{}),
	}
	runner := &runnerRef{
		llama:        persist,
		modelKey:     "model",
		gpus:         []ml.DeviceID{{Library: "CUDA"}},
		discreteGPUs: true,
	}
	s := InitScheduler(ctx)
	s.waitForRecovery = time.Millisecond
	s.loaded[runner.modelKey] = runner
	vramWaitStarted := make(chan struct{}, 1)
	s.getGpuFn = func(context.Context, []ml.FilteredRunnerDiscovery) []ml.DeviceInfo {
		select {
		case vramWaitStarted <- struct{}{}:
		default:
		}
		return nil
	}
	go s.processCompleted(ctx)
	s.expiredCh <- runner

	select {
	case <-persist.saveStarted:
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for prefill cache save")
	}

	lockAcquired := make(chan struct{})
	go func() {
		s.loadedMu.Lock()
		_ = len(s.loaded)
		s.loadedMu.Unlock()
		close(lockAcquired)
	}()
	select {
	case <-lockAcquired:
	case <-time.After(time.Second):
		t.Fatal("loadedMu remained locked during prefill cache save")
	}
	select {
	case <-vramWaitStarted:
		t.Fatal("VRAM recovery wait started before prefill cache save completed")
	default:
	}

	// Scheduler shutdown must cancel a save instead of waiting for its timeout.
	cancel()
	select {
	case <-vramWaitStarted:
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for VRAM recovery after prefill cache save")
	}
	select {
	case <-s.unloadedCh:
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for runner unload")
	}
}

func TestSchedPrefillCachePathSurvivesActiveLoadingRetry(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), time.Second)
	defer cancel()

	s := InitScheduler(ctx)
	s.prefillCacheRoot = t.TempDir()
	scenario := newScenarioRequest(t, ctx, "prefill-retry-test", 20, nil, map[ml.DeviceID]uint64{})
	persist := &prefillCacheMock{mockLlm: scenario.srv}
	var configuredPath string
	s.newServerFn = func(_ ml.SystemInfo, _ []ml.DeviceInfo, model string, _ *ggml.GGML, _, _ []string, _ api.Options, _ int, config llm.LlamaServerConfig) (llm.LlamaServer, error) {
		configuredPath = config.PrefillCachePath
		persist.modelPath = model
		return persist, nil
	}

	gpu := ml.DeviceInfo{DeviceID: ml.DeviceID{Library: "Metal"}, TotalMemory: 30, FreeMemory: 10}
	if needEvict := s.load(scenario.req, ml.SystemInfo{}, []ml.DeviceInfo{gpu}, true); !needEvict {
		t.Fatal("first load did not request eviction")
	}
	if configuredPath == "" {
		t.Fatal("configured prefill cache path is empty")
	}
	if got := scenario.req.prefillCachePath; got != configuredPath {
		t.Fatalf("active loading prefill cache path = %q, want %q", got, configuredPath)
	}

	gpu.FreeMemory = 30
	if needEvict := s.load(scenario.req, ml.SystemInfo{}, []ml.DeviceInfo{gpu}, true); needEvict {
		t.Fatal("second load unexpectedly requested eviction")
	}
	select {
	case err := <-scenario.req.errCh:
		t.Fatalf("load returned %v, want a runner", err)
	case runner := <-scenario.req.successCh:
		if got := runner.prefillCacheDir; got != configuredPath {
			t.Fatalf("runner prefill cache path = %q, want %q", got, configuredPath)
		}
	case <-ctx.Done():
		t.Fatal("timed out waiting for retried runner")
	}
}

func TestSchedRestoresPrefillCacheBeforeReturningRunner(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), time.Second)
	defer cancel()
	s := InitScheduler(ctx)
	scenario := newScenarioRequest(t, ctx, "prefill-test", 10, nil, map[ml.DeviceID]uint64{})
	persist := &prefillCacheMock{mockLlm: scenario.srv}
	s.newServerFn = func(_ ml.SystemInfo, _ []ml.DeviceInfo, model string, _ *ggml.GGML, _, _ []string, _ api.Options, _ int, _ llm.LlamaServerConfig) (llm.LlamaServer, error) {
		persist.modelPath = model
		return persist, nil
	}

	s.load(scenario.req, ml.SystemInfo{}, nil, false)
	select {
	case err := <-scenario.req.errCh:
		t.Fatalf("load returned %v, want a runner", err)
	case <-scenario.req.successCh:
		if want := []string{"restore"}; !slices.Equal(persist.calls, want) {
			t.Fatalf("runner calls = %v, want %v", persist.calls, want)
		}
	case <-ctx.Done():
		t.Fatal("timed out waiting for loaded runner")
	}
}

func TestSchedPrefillCacheFailureFallsBackAndSavesBeforeClose(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), time.Second)
	defer cancel()
	s := InitScheduler(ctx)
	scenario := newScenarioRequest(t, ctx, "prefill-fallback-test", 10, nil, map[ml.DeviceID]uint64{})
	persist := &prefillCacheMock{mockLlm: scenario.srv, restoreErr: errors.New("corrupt snapshot")}
	s.newServerFn = func(_ ml.SystemInfo, _ []ml.DeviceInfo, model string, _ *ggml.GGML, _, _ []string, _ api.Options, _ int, _ llm.LlamaServerConfig) (llm.LlamaServer, error) {
		persist.modelPath = model
		return persist, nil
	}

	s.load(scenario.req, ml.SystemInfo{}, nil, false)
	var runner *runnerRef
	select {
	case err := <-scenario.req.errCh:
		t.Fatalf("load returned %v, want a runner", err)
	case runner = <-scenario.req.successCh:
	case <-ctx.Done():
		t.Fatal("timed out waiting for loaded runner")
	}
	if runner == nil {
		t.Fatal("load returned a nil runner, want a fall back to a cold cache")
	}

	runner.refMu.Lock()
	runner.unload()
	runner.refMu.Unlock()
	if want := []string{"restore", "save", "close"}; !slices.Equal(persist.calls, want) {
		t.Fatalf("runner calls = %v, want %v", persist.calls, want)
	}
}
