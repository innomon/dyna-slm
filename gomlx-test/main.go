package main

import (
	"fmt"
	"log"
	"os"

	"github.com/gomlx/gomlx/backends"
	_ "github.com/gomlx/gomlx/backends/xla" // Registers XLA
)

func main() {
	// 1. Set the environment variable programmatically if you want to hardcode XLA
	// Otherwise, run with GOMLX_BACKEND=xla in your terminal
	os.Setenv("GOMLX_BACKEND", "xla")

	// 2. Updated v0.26.0 Signature: returns (Backend, error)
	backend, err := backends.New()
	if err != nil {
		log.Fatalf("Failed to create backend: %+v", err)
	}

	fmt.Printf("🚀 GoMLX successfully running on: %s\n", backend.Name())
	fmt.Printf("Available devices: %d\n", backend.NumDevices())
}
