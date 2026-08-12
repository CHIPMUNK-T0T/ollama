package cache

import (
	"fmt"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

// PersistentState is the serializable live state of one model cache layer.
// Arrays are stored separately in safetensors; the remaining fields are JSON
// metadata used to validate and reconstruct the concrete cache implementation.
type PersistentState struct {
	Kind   string       `json:"kind"`
	Offset int          `json:"offset"`
	Index  int          `json:"index,omitempty"`
	Arrays []*mlx.Array `json:"-"`
}

func ExportPersistentState(c Cache) (PersistentState, error) {
	if c == nil {
		return PersistentState{Kind: "nil"}, nil
	}

	state := c.State()
	if c.Offset() > 0 && len(state) < 2 {
		return PersistentState{}, fmt.Errorf("cache at offset %d has no state arrays", c.Offset())
	}
	if len(state) > 2 {
		state = state[:2]
	}

	switch c := c.(type) {
	case *KVCache:
		return PersistentState{Kind: "kv", Offset: c.offset, Arrays: state}, nil
	case *RotatingKVCache:
		return PersistentState{Kind: "rotating", Offset: c.offset, Index: c.idx, Arrays: state}, nil
	case *RecurrentCache:
		return PersistentState{Kind: "recurrent", Offset: c.offset, Arrays: state}, nil
	default:
		return PersistentState{}, fmt.Errorf("unsupported cache type %T", c)
	}
}

func ImportPersistentState(c Cache, state PersistentState) error {
	if c == nil {
		if state.Kind != "nil" {
			return fmt.Errorf("snapshot contains %s state for nil cache", state.Kind)
		}
		return nil
	}
	if state.Offset < 0 || (state.Offset > 0 && len(state.Arrays) != 2) {
		return fmt.Errorf("invalid %s cache state at offset %d", state.Kind, state.Offset)
	}
	if state.Offset == 0 {
		c.Free()
		return nil
	}
	for _, array := range state.Arrays {
		if array == nil || array.NumDims() < 3 {
			return fmt.Errorf("invalid %s cache tensor", state.Kind)
		}
	}

	switch c := c.(type) {
	case *KVCache:
		if state.Kind != "kv" || state.Arrays[0].Dim(2) < state.Offset || state.Arrays[1].Dim(2) < state.Offset {
			return fmt.Errorf("incompatible kv cache state")
		}
		c.Free()
		c.keys, c.values = state.Arrays[0], state.Arrays[1]
		c.offset = state.Offset
		mlx.Pin(c.keys, c.values)
	case *RotatingKVCache:
		if state.Kind != "rotating" || state.Index < 0 || state.Index > state.Arrays[0].Dim(2) {
			return fmt.Errorf("incompatible rotating cache state")
		}
		c.Free()
		c.keys, c.values = state.Arrays[0], state.Arrays[1]
		c.offset, c.idx = state.Offset, state.Index
		mlx.Pin(c.keys, c.values)
	case *RecurrentCache:
		if state.Kind != "recurrent" {
			return fmt.Errorf("incompatible recurrent cache state")
		}
		c.Free()
		c.convState, c.deltaState = state.Arrays[0], state.Arrays[1]
		c.offset = state.Offset
		mlx.Pin(c.convState, c.deltaState)
	default:
		return fmt.Errorf("unsupported cache type %T", c)
	}
	return nil
}
