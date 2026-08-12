package mlxrunner

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
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

func mustSave(t *testing.T, c *prefixCache, path string) {
	t.Helper()
	if err := c.save(t.Context(), path); err != nil {
		t.Fatalf("save(%s) = %v, want nil", path, err)
	}
}

func mustRestore(t *testing.T, c *prefixCache, path string) {
	t.Helper()
	if err := c.restore(t.Context(), path); err != nil {
		t.Fatalf("restore(%s) = %v, want nil", path, err)
	}
}

func wantOffset(t *testing.T, c *prefixCache, want int) {
	t.Helper()
	if got := c.minCacheOffset(); got != want {
		t.Fatalf("minCacheOffset() = %d, want %d", got, want)
	}
}

func wantRestoreError(t *testing.T, c *prefixCache, path, substr string) {
	t.Helper()
	err := c.restore(t.Context(), path)
	if err == nil || !strings.Contains(err.Error(), substr) {
		t.Fatalf("restore(%s) = %v, want an error containing %q", path, err, substr)
	}
}

func TestPrefixCacheCanceledRestoreKeepsSnapshot(t *testing.T) {
	dir := t.TempDir()
	snapshot := filepath.Join(dir, prefixCacheFilename)
	checksum := filepath.Join(dir, prefixCacheChecksumFilename)
	mustWriteFile(t, snapshot, "snapshot")
	mustWriteFile(t, checksum, "checksum")

	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	c := &prefixCache{}
	if err := c.restore(ctx, dir); !errors.Is(err, context.Canceled) {
		t.Fatalf("restore() = %v, want %v", err, context.Canceled)
	}
	mustExist(t, snapshot)
	mustExist(t, checksum)

	// Unload after an interrupted restore must not reinterpret the empty
	// in-memory cache as a request to invalidate the healthy snapshot.
	if err := c.save(t.Context(), dir); err != nil {
		t.Fatalf("save() = %v, want nil", err)
	}
	mustExist(t, snapshot)
	mustExist(t, checksum)
}

func TestPrefixCacheSafetensorsRoundTrip(t *testing.T) {
	skipIfNoMLX(t)

	input := &batch.Batch{
		InputIDs:     mlx.Zeros(mlx.DTypeInt32, 1, 3),
		SeqOffsets:   []int32{0},
		SeqQueryLens: []int32{3},
	}
	kv := cache.NewKVCache()
	keys := mlx.Zeros(mlx.DTypeFloat16, 1, 2, 3, 4)
	values := mlx.Zeros(mlx.DTypeFloat16, 1, 2, 3, 4)
	kv.Update(input, keys, values)

	rotating := cache.NewRotatingKVCache(16)
	rotating.Update(input, keys, values)

	recurrent := cache.NewRecurrentCache(2, 4, 2, 3, 4)
	recurrent.Get(input, mlx.DTypeFloat16)
	recurrent.Put(input,
		[]*mlx.Array{mlx.Zeros(mlx.DTypeFloat16, 1, 2, 4)},
		[]*mlx.Array{mlx.Zeros(mlx.DTypeFloat32, 1, 2, 3, 4)},
	)

	root := &trieNode{}
	leaf := &trieNode{
		tokens:    []trieKey{11, 22, 33},
		endOffset: 3,
		parent:    root,
	}
	root.children = []*trieNode{leaf}
	original := &prefixCache{
		root:       root,
		activePath: []*trieNode{root, leaf},
		caches:     []cache.Cache{kv, rotating, recurrent},
	}

	path := t.TempDir()
	mustSave(t, original, path)
	mustExist(t, filepath.Join(path, prefixCacheFilename))

	restoredKV := cache.NewKVCache()
	restoredRotating := cache.NewRotatingKVCache(16)
	restoredRecurrent := cache.NewRecurrentCache(2, 4, 2, 3, 4)
	restored := &prefixCache{
		caches: []cache.Cache{restoredKV, restoredRotating, restoredRecurrent},
	}
	mustRestore(t, restored, path)
	wantOffset(t, restored, 3)
	for name, offset := range map[string]int{
		"kv":        restoredKV.Offset(),
		"rotating":  restoredRotating.Offset(),
		"recurrent": restoredRecurrent.Offset(),
	} {
		if offset != 3 {
			t.Fatalf("%s cache offset = %d, want 3", name, offset)
		}
	}
	if len(restored.activePath) != 2 {
		t.Fatalf("activePath length = %d, want 2", len(restored.activePath))
	}
	if want := []trieKey{11, 22, 33}; !slices.Equal(restored.activePath[1].tokens, want) {
		t.Fatalf("restored tokens = %v, want %v", restored.activePath[1].tokens, want)
	}
	if restored.activePath[1].endOffset != 3 {
		t.Fatalf("restored endOffset = %d, want 3", restored.activePath[1].endOffset)
	}

	original.freeAll()
	restored.freeAll()
}

func TestPrefixCacheSaveAtomicallyReplacesPreviousSnapshot(t *testing.T) {
	skipIfNoMLX(t)

	makeCache := func(tokens []trieKey) *prefixCache {
		input := &batch.Batch{
			InputIDs:     mlx.Zeros(mlx.DTypeInt32, 1, len(tokens)),
			SeqOffsets:   []int32{0},
			SeqQueryLens: []int32{int32(len(tokens))},
		}
		kv := cache.NewKVCache()
		kv.Update(input,
			mlx.Zeros(mlx.DTypeFloat16, 1, 2, len(tokens), 4),
			mlx.Zeros(mlx.DTypeFloat16, 1, 2, len(tokens), 4),
		)
		root := &trieNode{}
		leaf := &trieNode{tokens: tokens, endOffset: len(tokens), parent: root}
		root.children = []*trieNode{leaf}
		return &prefixCache{
			root:       root,
			activePath: []*trieNode{root, leaf},
			caches:     []cache.Cache{kv},
		}
	}

	path := t.TempDir()
	first := makeCache([]trieKey{11, 22, 33})
	mustSave(t, first, path)

	second := makeCache([]trieKey{44, 55})
	mustSave(t, second, path)
	mustNotExist(t, filepath.Join(path, prefixCacheFilename+".tmp.safetensors"))

	restored := &prefixCache{caches: []cache.Cache{cache.NewKVCache()}}
	mustRestore(t, restored, path)
	wantOffset(t, restored, 2)
	if want := []trieKey{44, 55}; !slices.Equal(restored.activePath[1].tokens, want) {
		t.Fatalf("restored tokens = %v, want %v", restored.activePath[1].tokens, want)
	}

	first.freeAll()
	second.freeAll()
	restored.freeAll()
}

func TestPrefixCacheSaveUsesDeepestRestorableSnapshot(t *testing.T) {
	skipIfNoMLX(t)

	input3 := &batch.Batch{
		InputIDs:     mlx.Zeros(mlx.DTypeInt32, 1, 3),
		SeqOffsets:   []int32{0},
		SeqQueryLens: []int32{3},
	}
	input2 := &batch.Batch{
		InputIDs:     mlx.Zeros(mlx.DTypeInt32, 1, 2),
		SeqOffsets:   []int32{3},
		SeqQueryLens: []int32{2},
	}

	kv := cache.NewKVCache()
	kv.Update(input3,
		mlx.Zeros(mlx.DTypeFloat16, 1, 2, 3, 4),
		mlx.Zeros(mlx.DTypeFloat16, 1, 2, 3, 4),
	)
	recurrent := cache.NewRecurrentCache(2, 4, 2, 3, 4)
	recurrent.Get(input3, mlx.DTypeFloat16)
	recurrent.Put(input3,
		[]*mlx.Array{mlx.Zeros(mlx.DTypeFloat16, 1, 2, 4)},
		[]*mlx.Array{mlx.Zeros(mlx.DTypeFloat32, 1, 2, 3, 4)},
	)

	root := &trieNode{}
	prompt := &trieNode{
		tokens:    []trieKey{11, 22, 33},
		endOffset: 3,
		parent:    root,
		snapshots: []cache.Snapshot{kv.Snapshot(0), recurrent.Snapshot(0)},
	}
	generated := &trieNode{
		tokens:    []trieKey{44, 55},
		endOffset: 5,
		parent:    prompt,
	}
	root.children = []*trieNode{prompt}
	prompt.children = []*trieNode{generated}

	// Advance the live state beyond the reusable prompt snapshot. Recurrent
	// state cannot rewind from offset 5 without the captured offset-3 state.
	kv.Update(input2,
		mlx.Zeros(mlx.DTypeFloat16, 1, 2, 2, 4),
		mlx.Zeros(mlx.DTypeFloat16, 1, 2, 2, 4),
	)
	recurrent.Get(input2, mlx.DTypeFloat16)
	recurrent.Put(input2,
		[]*mlx.Array{mlx.Zeros(mlx.DTypeFloat16, 1, 2, 4)},
		[]*mlx.Array{mlx.Zeros(mlx.DTypeFloat32, 1, 2, 3, 4)},
	)
	if got := recurrent.Offset(); got != 5 {
		t.Fatalf("recurrent offset = %d, want 5", got)
	}

	original := &prefixCache{
		root:       root,
		activePath: []*trieNode{root, prompt, generated},
		caches:     []cache.Cache{kv, recurrent},
	}
	path := t.TempDir()
	mustSave(t, original, path)
	wantOffset(t, original, 3)

	restored := &prefixCache{
		caches: []cache.Cache{cache.NewKVCache(), cache.NewRecurrentCache(2, 4, 2, 3, 4)},
	}
	mustRestore(t, restored, path)
	wantOffset(t, restored, 3)
	if !restored.activePath[1].user {
		t.Fatal("restored boundary node is not marked user, want it to resist auto-merge")
	}
	if !hasAllSnapshots(restored.activePath[1], restored.caches) {
		t.Fatal("restored boundary node is missing snapshots, want a complete set")
	}

	session := restored.begin([]int32{11, 22, 33, 44}, nil)
	if want := []int32{44}; !slices.Equal(session.remaining, want) {
		t.Fatalf("session remaining = %v, want %v", session.remaining, want)
	}

	// Advance the restored runner again and persist it a second time. The
	// restore boundary must survive this cycle, or only the first reload is
	// warm for recurrent models.
	restoredKV := restored.caches[0].(*cache.KVCache)
	restoredRecurrent := restored.caches[1].(*cache.RecurrentCache)
	restoredKV.Update(input2,
		mlx.Zeros(mlx.DTypeFloat16, 1, 2, 2, 4),
		mlx.Zeros(mlx.DTypeFloat16, 1, 2, 2, 4),
	)
	restoredRecurrent.Get(input2, mlx.DTypeFloat16)
	restoredRecurrent.Put(input2,
		[]*mlx.Array{mlx.Zeros(mlx.DTypeFloat16, 1, 2, 4)},
		[]*mlx.Array{mlx.Zeros(mlx.DTypeFloat32, 1, 2, 3, 4)},
	)
	restoredGenerated := &trieNode{
		tokens:    []trieKey{44, 55},
		endOffset: 5,
		parent:    restored.activePath[1],
	}
	restored.activePath[1].children = []*trieNode{restoredGenerated}
	restored.activePath = append(restored.activePath, restoredGenerated)

	secondPath := t.TempDir()
	mustSave(t, restored, secondPath)
	wantOffset(t, restored, 3)

	restoredAgain := &prefixCache{
		caches: []cache.Cache{cache.NewKVCache(), cache.NewRecurrentCache(2, 4, 2, 3, 4)},
	}
	mustRestore(t, restoredAgain, secondPath)
	wantOffset(t, restoredAgain, 3)
	secondSession := restoredAgain.begin([]int32{11, 22, 33, 44}, nil)
	if want := []int32{44}; !slices.Equal(secondSession.remaining, want) {
		t.Fatalf("second session remaining = %v, want %v", secondSession.remaining, want)
	}

	original.freeAll()
	restored.freeAll()
	restoredAgain.freeAll()
}

func TestPrefixCacheShortRecurrentPersistFallsBackCold(t *testing.T) {
	skipIfNoMLX(t)

	input := &batch.Batch{
		InputIDs:     mlx.Zeros(mlx.DTypeInt32, 1, 3),
		SeqOffsets:   []int32{0},
		SeqQueryLens: []int32{3},
	}
	recurrent := cache.NewRecurrentCache(2, 4, 2, 3, 4)
	recurrent.Get(input, mlx.DTypeFloat16)
	recurrent.Put(input,
		[]*mlx.Array{mlx.Zeros(mlx.DTypeFloat16, 1, 2, 4)},
		[]*mlx.Array{mlx.Zeros(mlx.DTypeFloat32, 1, 2, 3, 4)},
	)

	root := &trieNode{}
	leaf := &trieNode{
		tokens:    []trieKey{11, 22, 33},
		endOffset: 3,
		parent:    root,
	}
	root.children = []*trieNode{leaf}
	original := &prefixCache{
		root:       root,
		activePath: []*trieNode{root, leaf},
		caches:     []cache.Cache{recurrent},
	}

	path := t.TempDir()
	mustSave(t, original, path)

	restored := &prefixCache{
		caches: []cache.Cache{cache.NewRecurrentCache(2, 4, 2, 3, 4)},
	}
	mustRestore(t, restored, path)
	wantOffset(t, restored, 3)

	// This models a prompt shorter than the persisted prompt+generation
	// frontier. Recurrent state cannot rewind to the partial match, so begin
	// must discard it and run the complete short prompt cold.
	session := restored.begin([]int32{11, 22}, nil)
	wantOffset(t, restored, 0)
	if want := []int32{11, 22}; !slices.Equal(session.remaining, want) {
		t.Fatalf("session remaining = %v, want %v", session.remaining, want)
	}

	original.freeAll()
	restored.freeAll()
}

func TestPrefixCacheRotatingWindowPersistFromRestorableBoundary(t *testing.T) {
	skipIfNoMLX(t)

	updateOne := func(c *cache.RotatingKVCache, offset int) {
		input := &batch.Batch{
			InputIDs:     mlx.Zeros(mlx.DTypeInt32, 1, 1),
			SeqOffsets:   []int32{int32(offset)},
			SeqQueryLens: []int32{1},
		}
		c.Update(input,
			mlx.Zeros(mlx.DTypeFloat16, 1, 2, 1, 4),
			mlx.Zeros(mlx.DTypeFloat16, 1, 2, 1, 4),
		)
	}

	rotating := cache.NewRotatingKVCache(4)
	for i := range 6 {
		updateOne(rotating, i)
	}
	promptSnapshot := rotating.Snapshot(0)
	for i := 6; i < 8; i++ {
		updateOne(rotating, i)
	}
	if got := rotating.Offset(); got != 8 {
		t.Fatalf("rotating offset = %d, want 8", got)
	}

	root := &trieNode{}
	prompt := &trieNode{
		tokens:    []trieKey{1, 2, 3, 4, 5, 6},
		endOffset: 6,
		parent:    root,
		snapshots: []cache.Snapshot{promptSnapshot},
	}
	generated := &trieNode{
		tokens:    []trieKey{7, 8},
		endOffset: 8,
		parent:    prompt,
	}
	root.children = []*trieNode{prompt}
	prompt.children = []*trieNode{generated}
	original := &prefixCache{
		root:       root,
		activePath: []*trieNode{root, prompt, generated},
		caches:     []cache.Cache{rotating},
	}

	path := t.TempDir()
	mustSave(t, original, path)
	wantOffset(t, original, 6)

	restoredRotating := cache.NewRotatingKVCache(4)
	restored := &prefixCache{caches: []cache.Cache{restoredRotating}}
	mustRestore(t, restored, path)
	if got := restoredRotating.Offset(); got != 6 {
		t.Fatalf("restored rotating offset = %d, want 6", got)
	}
	state, err := cache.ExportPersistentState(restoredRotating)
	if err != nil {
		t.Fatalf("ExportPersistentState() = %v, want nil", err)
	}
	// The ring index and window width must survive the round trip, or the
	// restored window reads its entries in the wrong order.
	if state.Index != 2 {
		t.Fatalf("ring index = %d, want 2", state.Index)
	}
	if got := state.Arrays[0].Dim(2); got != 4 {
		t.Fatalf("window width = %d, want 4", got)
	}

	session := restored.begin([]int32{1, 2, 3, 4, 5, 6, 9}, nil)
	if want := []int32{9}; !slices.Equal(session.remaining, want) {
		t.Fatalf("session remaining = %v, want %v", session.remaining, want)
	}

	original.freeAll()
	restored.freeAll()
}

func TestPrefixCacheMediaFoldSurvivesPersistence(t *testing.T) {
	skipIfNoMLX(t)

	tokens := []int32{1, 2, 900, 900, 900, 3, 4}
	mediaA := []mediaItem{{pos: 2, length: 3, fold: foldValue([]byte("a"), []int{1})}}
	mediaB := []mediaItem{{pos: 2, length: 3, fold: foldValue([]byte("b"), []int{1})}}

	input := &batch.Batch{
		InputIDs:     mlx.Zeros(mlx.DTypeInt32, 1, len(tokens)),
		SeqOffsets:   []int32{0},
		SeqQueryLens: []int32{int32(len(tokens))},
	}
	kv := cache.NewKVCache()
	kv.Update(input,
		mlx.Zeros(mlx.DTypeFloat16, 1, 2, len(tokens), 4),
		mlx.Zeros(mlx.DTypeFloat16, 1, 2, len(tokens), 4),
	)

	original := &prefixCache{caches: []cache.Cache{kv}}
	original.ensureRoot()
	keys := original.key(effectiveKeyTokens(tokens, mediaA))
	leaf := &trieNode{tokens: keys, endOffset: len(keys), parent: original.root}
	original.root.children = []*trieNode{leaf}
	original.activePath = []*trieNode{original.root, leaf}

	path := t.TempDir()
	mustSave(t, original, path)

	restoredSame := &prefixCache{caches: []cache.Cache{cache.NewKVCache()}}
	mustRestore(t, restoredSame, path)
	sameSession := restoredSame.begin(append(tokens, 5), mediaA)
	if want := []int32{5}; !slices.Equal(sameSession.remaining, want) {
		t.Fatalf("same-media session remaining = %v, want %v", sameSession.remaining, want)
	}

	// Different media behind identical placeholder tokens must not match: the
	// fold is what distinguishes them.
	restoredDifferent := &prefixCache{caches: []cache.Cache{cache.NewKVCache()}}
	mustRestore(t, restoredDifferent, path)
	differentSession := restoredDifferent.begin(append(tokens, 5), mediaB)
	wantOffset(t, restoredDifferent, 2)
	if want := append(tokens[2:], 5); !slices.Equal(differentSession.remaining, want) {
		t.Fatalf("other-media session remaining = %v, want %v", differentSession.remaining, want)
	}

	original.freeAll()
	restoredSame.freeAll()
	restoredDifferent.freeAll()
}

func TestPrefixCacheRestoreMissingSnapshotIsColdMiss(t *testing.T) {
	c := &prefixCache{caches: []cache.Cache{cache.NewKVCache()}}
	mustRestore(t, c, t.TempDir())
	wantOffset(t, c, 0)
}

func TestPrefixCacheRestoreCorruptSnapshotFailsOpen(t *testing.T) {
	skipIfNoMLX(t)

	dir := t.TempDir()
	snapshot := filepath.Join(dir, prefixCacheFilename)
	mustWriteFile(t, snapshot, "not a safetensors file")
	// Checksum the garbage so verification passes and the safetensors load is
	// what rejects it: this covers structural damage, not payload corruption.
	if err := writePrefixCacheChecksum(t.Context(), dir, snapshot); err != nil {
		t.Fatalf("writePrefixCacheChecksum() = %v, want nil", err)
	}

	c := &prefixCache{caches: []cache.Cache{cache.NewKVCache()}}
	if err := c.restore(t.Context(), dir); err == nil {
		t.Fatal("restore() = nil, want an error")
	}
	// A failed restore invalidates the snapshot so the next load is a clean
	// cold miss.
	mustNotExist(t, snapshot)
	mustNotExist(t, filepath.Join(dir, prefixCacheChecksumFilename))
	wantOffset(t, c, 0)
}

func TestPrefixCacheChecksumCatchesPayloadCorruption(t *testing.T) {
	skipIfNoMLX(t)

	tokens := []trieKey{11, 22, 33}
	input := &batch.Batch{
		InputIDs:     mlx.Zeros(mlx.DTypeInt32, 1, len(tokens)),
		SeqOffsets:   []int32{0},
		SeqQueryLens: []int32{int32(len(tokens))},
	}
	kv := cache.NewKVCache()
	kv.Update(input,
		mlx.Zeros(mlx.DTypeFloat16, 1, 2, len(tokens), 4),
		mlx.Zeros(mlx.DTypeFloat16, 1, 2, len(tokens), 4),
	)
	root := &trieNode{}
	leaf := &trieNode{tokens: tokens, endOffset: len(tokens), parent: root}
	root.children = []*trieNode{leaf}
	original := &prefixCache{
		root:       root,
		activePath: []*trieNode{root, leaf},
		caches:     []cache.Cache{kv},
	}

	dir := t.TempDir()
	mustSave(t, original, dir)
	snapshot := filepath.Join(dir, prefixCacheFilename)

	// Flip the final byte. safetensors stores the header up front and tensor
	// data after it, so this lands in the payload: the file stays structurally
	// valid and every shape, dtype, and offset still agrees with its size.
	data, err := os.ReadFile(snapshot)
	if err != nil {
		t.Fatalf("read %s: %v", snapshot, err)
	}
	data[len(data)-1] ^= 0xff
	if err := os.WriteFile(snapshot, data, 0o600); err != nil {
		t.Fatalf("write %s: %v", snapshot, err)
	}

	// Structural validation alone accepts it — this is the gap the checksum
	// closes, and the reason a corrupt-snapshot drill that damages the header
	// does not prove the payload case.
	file, err := mlx.LoadSafetensorsNative(snapshot)
	if err != nil {
		t.Fatalf("LoadSafetensorsNative() = %v, want nil: payload corruption must survive structural validation", err)
	}
	file.Free()

	restored := &prefixCache{caches: []cache.Cache{cache.NewKVCache()}}
	wantRestoreError(t, restored, dir, "checksum mismatch")
	wantOffset(t, restored, 0)

	original.freeAll()
	restored.freeAll()
}

func TestPrefixCacheRestoreChecksumMismatchIsRefused(t *testing.T) {
	dir := t.TempDir()
	snapshot := filepath.Join(dir, prefixCacheFilename)
	mustWriteFile(t, snapshot, "payload")
	if err := writePrefixCacheChecksum(t.Context(), dir, snapshot); err != nil {
		t.Fatalf("writePrefixCacheChecksum() = %v, want nil", err)
	}
	mustWriteFile(t, snapshot, "payloae")

	// Verification runs before the snapshot reaches MLX, so this rejection is
	// exercisable without an MLX runtime.
	c := &prefixCache{caches: []cache.Cache{cache.NewKVCache()}}
	wantRestoreError(t, c, dir, "checksum mismatch")
	mustNotExist(t, snapshot)
	wantOffset(t, c, 0)
}

func TestPrefixCacheRestoreMissingChecksumIsColdMiss(t *testing.T) {
	dir := t.TempDir()
	snapshot := filepath.Join(dir, prefixCacheFilename)
	mustWriteFile(t, snapshot, "payload")

	// A save killed between committing the snapshot and committing its checksum
	// leaves an unverifiable snapshot. That is a cold miss, not an error: the
	// snapshot never reaches MLX, so nothing can be corrupted by it.
	c := &prefixCache{caches: []cache.Cache{cache.NewKVCache()}}
	mustRestore(t, c, dir)
	wantOffset(t, c, 0)
}

func TestPrefixCacheSaveEmptyInvalidatesChecksumToo(t *testing.T) {
	dir := t.TempDir()
	snapshot := filepath.Join(dir, prefixCacheFilename)
	mustWriteFile(t, snapshot, "stale")
	if err := writePrefixCacheChecksum(t.Context(), dir, snapshot); err != nil {
		t.Fatalf("writePrefixCacheChecksum() = %v, want nil", err)
	}

	// A stale checksum left next to a removed snapshot would validate whatever
	// happened to be written there next.
	c := &prefixCache{}
	mustSave(t, c, dir)
	mustNotExist(t, snapshot)
	mustNotExist(t, filepath.Join(dir, prefixCacheChecksumFilename))
}

func TestPrefixCacheRestoreRefusesWarmCache(t *testing.T) {
	dir := t.TempDir()
	snapshot := filepath.Join(dir, prefixCacheFilename)
	mustWriteFile(t, snapshot, "stale")

	c := &prefixCache{}
	c.ensureRoot()
	wantRestoreError(t, c, dir, "warm cache")
	// Caller state is not evidence that the on-disk snapshot is damaged.
	mustExist(t, snapshot)
}
