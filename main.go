package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"gopkg.in/ini.v1"

	flag "github.com/spf13/pflag"
)

const (
	Version      = "0.1.0"
	DefaultModel = "Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M"
)

type OllamaGenerateRequest struct {
	Model   string                 `json:"model"`
	Prompt  string                 `json:"prompt"`
	Stream  bool                   `json:"stream"`
	System  string                 `json:"system,omitempty"`
	Options map[string]interface{} `json:"options,omitempty"`
}

type OllamaPullRequest struct {
	Model string `json:"model"`
}

type OpenAIChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type StreamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type LlamaUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
}

type LlamaTimings struct {
	PromptMS    float64 `json:"prompt_ms"`
	PredictedMS float64 `json:"predicted_ms"`
}

type OpenAIChatChunk struct {
	Choices []struct {
		Delta struct {
			Role             string `json:"role"`
			Content          string `json:"content"`
			ReasoningContent string `json:"reasoning_content"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage   LlamaUsage   `json:"usage"`
	Timings LlamaTimings `json:"timings"`
}

type OllamaTagsResponse struct {
	Models []OllamaModelInfo `json:"models"`
}

type OllamaModelInfo struct {
	Name       string                 `json:"name"`
	Model      string                 `json:"model"`
	ModifiedAt time.Time              `json:"modified_at"`
	Size       int64                  `json:"size"`
	Digest     string                 `json:"digest"`
	Details    map[string]interface{} `json:"details"`
	RealPath   string                 `json:"-"` // internal only - not serialized to JSON
}

type OpenAIModelsResponse struct {
	Object string             `json:"object"`
	Data   []OpenAIModelPrice `json:"data"`
}

type OpenAIModelPrice struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

type OllamaPsResponse struct {
	Models []map[string]interface{} `json:"models"`
}

type OllamaVersionResponse struct {
	Version string `json:"version"`
}

type LlamaServerManager struct {
	sync.Mutex
	cmd       *exec.Cmd
	serverBin string
	args      []string
}

type RouterModelStatus struct {
	Value string `json:"value"`
}

type RouterModelMeta struct {
	NParams int64 `json:"n_params"`
	Size    int64 `json:"size"`
}

type RouterModel struct {
	ID     string            `json:"id"`
	Status RouterModelStatus `json:"status"`
	Meta   *RouterModelMeta  `json:"meta,omitempty"`
}

type RouterModelsResponse struct {
	Data []RouterModel `json:"data"`
}

type ModelPresetManager struct {
	sync.Mutex
	path string
}

// ---------------------------------------------------------------------------
// RPC Worker Discovery
// ---------------------------------------------------------------------------

const discoveryDefaultPort = 50051

// HeartbeatPacket is the UDP payload broadcast by worker nodes every second.
type HeartbeatPacket struct {
	Type    string `json:"type"`
	Addr    string `json:"addr"`
	Version string `json:"version"`
}

// RPCPeerRegistry is a thread-safe map of worker address -> last-seen time.
// Entries older than peerTimeout are pruned automatically by the reaper goroutine.
type RPCPeerRegistry struct {
	sync.RWMutex
	peers       map[string]time.Time
	peerTimeout time.Duration
}

func NewRPCPeerRegistry(timeout time.Duration) *RPCPeerRegistry {
	return &RPCPeerRegistry{
		peers:       make(map[string]time.Time),
		peerTimeout: timeout,
	}
}

// Touch records (or refreshes) a peer address.
// Touch records (or refreshes) a peer address. Returns true the first time an
// address is seen, false on subsequent refreshes — useful for log dedup.
func (r *RPCPeerRegistry) Touch(addr string) bool {
	r.Lock()
	defer r.Unlock()
	_, seen := r.peers[addr]
	r.peers[addr] = time.Now()
	return !seen
}

// Reap removes peers that have not sent a heartbeat within peerTimeout.
// Returns the number of peers pruned.
func (r *RPCPeerRegistry) Reap() int {
	r.Lock()
	defer r.Unlock()
	pruned := 0
	for addr, last := range r.peers {
		if time.Since(last) > r.peerTimeout {
			fmt.Printf("[DISCOVERY] Worker timed out, removing: %s\n", addr)
			delete(r.peers, addr)
			pruned++
		}
	}
	return pruned
}

// List returns a snapshot of all live peer addresses.
func (r *RPCPeerRegistry) List() []string {
	r.RLock()
	defer r.RUnlock()
	addrs := make([]string, 0, len(r.peers))
	for addr := range r.peers {
		addrs = append(addrs, addr)
	}
	return addrs
}

// outboundIP returns the IP address this host will use as the source of UDP
// broadcast packets. Routing toward 255.255.255.255 causes the kernel to
// select the LAN-facing interface (e.g. eth0 / 192.168.x.x) rather than a
// Docker bridge interface (172.17.x.x / 172.19.x.x), which is what we want
// so that other hosts on the same LAN can actually reach this worker.
// No packets are sent — this is purely a routing-table query.
func outboundIP() string {
	conn, err := net.Dial("udp4", "255.255.255.255:50051")
	if err != nil {
		// Fallback: try a well-known external address; on plain hosts (no Docker)
		// this also gives the right answer.
		conn2, err2 := net.Dial("udp4", "8.8.8.8:80")
		if err2 != nil {
			return "127.0.0.1"
		}
		defer conn2.Close()
		return conn2.LocalAddr().(*net.UDPAddr).IP.String()
	}
	defer conn.Close()
	return conn.LocalAddr().(*net.UDPAddr).IP.String()
}

// broadcastHeartbeat is run as a goroutine on worker nodes when --discovery is
// active. It sends a UDP broadcast packet every second until stop is closed.
// Requires network_mode: host (or equivalent) so the socket reaches the
// physical LAN interface rather than a Docker bridge.
func broadcastHeartbeat(rpcAddr string, discoveryPort int, stop <-chan struct{}) {
	dst, err := net.ResolveUDPAddr("udp4", fmt.Sprintf("255.255.255.255:%d", discoveryPort))
	if err != nil {
		fmt.Printf("[DISCOVERY] Cannot resolve broadcast address: %v\n", err)
		return
	}
	conn, err := net.DialUDP("udp4", nil, dst)
	if err != nil {
		fmt.Printf("[DISCOVERY] Cannot open UDP socket for heartbeat: %v\n", err)
		return
	}
	defer conn.Close()

	pkt, _ := json.Marshal(HeartbeatPacket{
		Type:    "rpc-heartbeat",
		Addr:    rpcAddr,
		Version: Version,
	})

	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()
	fmt.Printf("[DISCOVERY] Broadcasting heartbeat for %s on port %d\n", rpcAddr, discoveryPort)
	for {
		select {
		case <-stop:
			return
		case <-ticker.C:
			if _, err := conn.Write(pkt); err != nil {
				fmt.Printf("[DISCOVERY] Heartbeat send error: %v\n", err)
			}
		}
	}
}

// listenForWorkers is run as a goroutine on serve nodes when --discovery is
// active. It listens for UDP broadcast heartbeats from worker nodes.
// Requires network_mode: host so broadcast packets from other machines arrive
// on the physical interface rather than being confined to a Docker bridge.
func listenForWorkers(registry *RPCPeerRegistry, discoveryPort int, stop <-chan struct{}) {
	addr := fmt.Sprintf("0.0.0.0:%d", discoveryPort)
	pc, err := net.ListenPacket("udp4", addr)
	if err != nil {
		fmt.Printf("[DISCOVERY] Cannot listen on UDP %s: %v\n", addr, err)
		return
	}
	defer pc.Close()

	go func() {
		<-stop
		pc.Close()
	}()

	fmt.Printf("[DISCOVERY] Listening for broadcast heartbeats on %s\n", addr)
	buf := make([]byte, 1024)
	for {
		n, _, err := pc.ReadFrom(buf)
		if err != nil {
			// stop was triggered — exit cleanly
			return
		}
		var pkt HeartbeatPacket
		if err := json.Unmarshal(buf[:n], &pkt); err != nil {
			continue
		}
		if pkt.Type != "rpc-heartbeat" || pkt.Addr == "" {
			continue
		}
		if registry.Touch(pkt.Addr) {
			fmt.Printf("[DISCOVERY] New worker found: %s (version %s)\n", pkt.Addr, pkt.Version)
		}
	}
}

// startPeerReaper runs a background goroutine that calls registry.Reap() every
// second. Returns a stop channel the caller closes to halt the reaper.
func startPeerReaper(registry *RPCPeerRegistry) chan struct{} {
	stop := make(chan struct{})
	go func() {
		ticker := time.NewTicker(1 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-stop:
				return
			case <-ticker.C:
				registry.Reap()
			}
		}
	}()
	return stop
}

// sortedJoin sorts a slice of strings and joins them with commas.
// Used to produce a stable --rpc argument so we can detect real roster changes
// without being fooled by map iteration order differences.
func sortedJoin(addrs []string) string {
	sorted := make([]string, len(addrs))
	copy(sorted, addrs)
	for i := 1; i < len(sorted); i++ {
		for j := i; j > 0 && sorted[j] < sorted[j-1]; j-- {
			sorted[j], sorted[j-1] = sorted[j-1], sorted[j]
		}
	}
	return strings.Join(sorted, ",")
}

// startDiscoveryWatcher starts the UDP listener and reaper, waits initialWait
// for the first heartbeats to arrive, then launches llama-server with the
// discovered workers. It continues running in the background: whenever the
// live worker roster changes (new worker appears or one times out) it updates
// llamaServer's args and restarts it automatically.
//
// baseArgs are the llama-server flags without --rpc (port, models-dir, preset,
// any extra passthrough args). The watcher splices --rpc in/out as needed.
func startDiscoveryWatcher(initialWait time.Duration, discoveryPort int, baseArgs []string, llamaServer *LlamaServerManager, waitForReady func()) {
	registry := NewRPCPeerRegistry(5 * time.Second)

	stopListener := make(chan struct{})
	go listenForWorkers(registry, discoveryPort, stopListener)
	_ = stopListener // listener runs for the lifetime of the process

	stopReaper := startPeerReaper(registry)
	_ = stopReaper // reaper runs for the lifetime of the process

	// Give the listener a moment to bind, then wait for the initial scan window.
	time.Sleep(50 * time.Millisecond)
	fmt.Printf("[DISCOVERY] Listening for workers for %v...\n", initialWait)
	time.Sleep(initialWait)

	// Start llama-server with whatever workers showed up during the initial wait.
	currentRoster := sortedJoin(registry.List())
	llamaServer.SetArgs(baseArgs, currentRoster)
	if currentRoster != "" {
		fmt.Printf("[SERVE] Using discovered RPC workers: %s\n", currentRoster)
	} else {
		fmt.Println("[SERVE] No workers found, starting in single-node mode")
	}
	if err := llamaServer.Start(); err != nil {
		fmt.Println("[ERROR] Failed to start llama-server:", err)
		os.Exit(1)
	}
	waitForReady()

	// Background loop: check for roster changes every second and restart if needed.
	go func() {
		for {
			time.Sleep(1 * time.Second)
			newRoster := sortedJoin(registry.List())
			if newRoster == currentRoster {
				continue
			}
			if newRoster == "" {
				fmt.Println("[DISCOVERY] All workers lost, restarting in single-node mode")
			} else if currentRoster == "" {
				fmt.Printf("[DISCOVERY] Workers joined: %s\n", newRoster)
			} else {
				fmt.Printf("[DISCOVERY] Worker roster changed: %s -> %s\n", currentRoster, newRoster)
			}
			currentRoster = newRoster
			llamaServer.SetArgs(baseArgs, currentRoster)
			if err := llamaServer.RestartWithReason("worker roster changed"); err != nil {
				fmt.Println("[ERROR] Failed to restart llama-server after roster change:", err)
			}
			waitForReady()
		}
	}()
}

// PullProgress is reported during a model download via downloadModel's
// callback, letting callers (HTTP handler, CLI) render it however they like.
type PullProgress struct {
	Status    string
	Digest    string
	Completed int64
	Total     int64
	Done      bool
	Error     error
}

func main() {
	portFlag := flag.IntP("port", "", 11434, "Specify API/RPC port")
	llamaPortFlag := flag.IntP("llama-port", "", 8080, "Specify Llama.cpp API port")
	rpcPortFlag := flag.IntP("rpc-port", "", 50052, "Specify RPC port")
	modelFlag := flag.StringP("model", "m", DefaultModel, "Specified GGUF model")
	verboseFlag := flag.BoolP("verbose", "v", false, "Verbose mode")

	// Manual RPC flag (serve): skip discovery and connect to explicit workers.
	// Example: --rpc 192.168.0.120:50052,192.168.0.110:50052
	rpcFlag := flag.StringP("rpc", "", "", "Comma-separated list of RPC worker addresses (host:port); disables discovery")

	// Discovery flags
	discoveryFlag := flag.BoolP("discovery", "", false, "Auto-discover RPC workers via UDP heartbeat (serve & worker)")
	discoveryPortFlag := flag.IntP("discovery-port", "", discoveryDefaultPort, "UDP port used for worker heartbeat discovery")
	discoveryWaitFlag := flag.DurationP("discovery-wait", "", 8*time.Second, "How long serve waits for worker heartbeats before starting llama-server")
	// Explicit advertise address for worker nodes running inside Docker / NAT,
	// where the auto-detected IP is a container bridge address rather than the
	// host's LAN IP.  Example: --advertise-addr 192.168.0.120
	advertiseAddrFlag := flag.StringP("advertise-addr", "", "", "IP (or host:port) to advertise in heartbeats; overrides auto-detection (useful inside Docker)")

	flag.Parse()

	fmt.Printf("[INFO] Starting LlamaNexus version %s\n", Version)

	port := *portFlag
	llamaport := *llamaPortFlag
	rpcport := *rpcPortFlag
	verbose := *verboseFlag
	model := *modelFlag
	rpcAddrs := *rpcFlag
	discovery := *discoveryFlag
	discoveryPort := *discoveryPortFlag
	discoveryWait := *discoveryWaitFlag
	advertiseAddr := *advertiseAddrFlag

	args := flag.Args()
	command := ""
	if len(args) > 0 {
		command = args[0]
	} else {
		flag.Usage()
		os.Exit(1)
	}

	if command == "serve" {
		fmt.Printf("[SERVE] Starting llama-server: port=%d llama port=%d verbose=%t\n", port, llamaport, verbose)
		runServe(port, llamaport, verbose, rpcAddrs, discovery, discoveryPort, discoveryWait, args[1:])
	} else if command == "run" {
		fmt.Printf("[RUN] Starting llama-cli: model=%s verbose=%t\n", model, verbose)
		runCliInference(model, verbose, args[1:])
	} else if command == "pull" {
		if len(args) < 2 {
			fmt.Println("[ERROR] usage: llamanexus pull <repo:tag>")
			os.Exit(1)
		}
		runPull(args[1], verbose)
	} else if command == "worker" {
		fmt.Printf("[WORKER] Starting worker: port=%d discovery=%t\n", rpcport, discovery)
		runRpcServer(rpcport, discovery, discoveryPort, advertiseAddr, args[1:])
	} else {
		flag.Usage()
		os.Exit(1)
	}
}

func NewModelPresetManager(path string) *ModelPresetManager {
	return &ModelPresetManager{path: path}
}

// LoadCtxSizes reads the preset file (if it exists) and returns a map of
// modelID -> ctx-size for every section that explicitly sets one. Used at
// startup to seed CtxSizeTracker with real, persisted state.
func (m *ModelPresetManager) LoadCtxSizes() map[string]int {
	fmt.Printf("[PRESET] Loading ctx-size overrides from %s\n", m.path)

	m.Lock()
	defer m.Unlock()

	result := make(map[string]int)
	cfg, err := ini.Load(m.path)
	if err != nil {
		return result // file doesn't exist yet - nothing overridden, that's fine
	}

	for _, section := range cfg.Sections() {
		name := section.Name()
		if name == "DEFAULT" || name == "*" {
			continue // global defaults section, not a per-model override
		}
		if section.HasKey("ctx-size") {
			if val, err := section.Key("ctx-size").Int(); err == nil {
				result[name] = val
			}
		}
	}
	return result
}

// SetCtxSize writes/updates the ctx-size override for a specific model section,
// preserving any other keys already present in that section or elsewhere
// in the file.
func (m *ModelPresetManager) SetCtxSize(modelID string, ctxSize int) error {
	fmt.Printf("[PRESET] Setting ctx-size for model %s in file %s\n", modelID, m.path)

	m.Lock()
	defer m.Unlock()

	cfg, err := ini.LoadSources(ini.LoadOptions{AllowShadows: true}, m.path)
	if err != nil {
		fmt.Printf("[CTX] preset file %s doesn't exist yet, creating new one\n", m.path)
		cfg = ini.Empty() // file doesn't exist yet - start fresh
	}

	section, err := cfg.GetSection(modelID)
	if err != nil {
		section, err = cfg.NewSection(modelID)
		if err != nil {
			return fmt.Errorf("failed to create preset section for %s: %w", modelID, err)
		}
	}
	section.Key("ctx-size").SetValue(strconv.Itoa(ctxSize))

	return cfg.SaveTo(m.path)
}

// ctxSizeTracker remembers the last ctx-size we applied per model, so we only
// rewrite the preset file and trigger an unload/reload when Open WebUI's
// num_ctx value actually changes - not on every single chat message.
type CtxSizeTracker struct {
	sync.RWMutex
	current map[string]int // modelID -> last-applied ctx-size
}

func NewCtxSizeTracker() *CtxSizeTracker {
	return &CtxSizeTracker{current: make(map[string]int)}
}

// NeedsUpdate reports whether requestedCtx differs from what we last applied
// for this model. Returns true if this is the first time we've seen this
// model (no recorded value yet) AND requestedCtx differs from the server's
// own default - see note below about needing the real default for that case.
func (t *CtxSizeTracker) NeedsUpdate(modelID string, requestedCtx int) bool {
	t.RLock()
	defer t.RUnlock()
	last, seen := t.current[modelID]
	if !seen {
		return true // never tracked before - worth checking against default
	}
	return last != requestedCtx
}

func (t *CtxSizeTracker) Record(modelID string, ctxSize int) {
	t.Lock()
	defer t.Unlock()
	t.current[modelID] = ctxSize
}

// EnsureGlobalDefault writes the [*] global section with a default ctx-size,
// if one isn't already present, ensuring the router always has a baseline
// even before any per-model override is ever applied.
func (m *ModelPresetManager) EnsureGlobalDefault(defaultCtxSize int) error {
	fmt.Printf("[PRESET] Checking custom preset file %s\n", m.path)

	m.Lock()
	defer m.Unlock()

	cfg, err := ini.Load(m.path)
	if err != nil {
		fmt.Printf("[PRESET] preset file %s doesn't exist yet, creating new one\n", m.path)
		cfg = ini.Empty()
	}

	section, err := cfg.GetSection("*")
	if err != nil {
		fmt.Printf("[PRESET] Create global section for file %s\n", m.path)
		section, err = cfg.NewSection("*")
		if err != nil {
			return err
		}
	}
	if !section.HasKey("ctx-size") {
		fmt.Printf("[PRESET] Create ctx-size for global section in file %s\n", m.path)
		section.Key("ctx-size").SetValue(strconv.Itoa(defaultCtxSize))
	}

	return cfg.SaveTo(m.path)
}

func (m *LlamaServerManager) Start() error {
	m.Lock()
	defer m.Unlock()
	cmd := exec.Command(m.serverBin, m.args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		return err
	}
	m.cmd = cmd
	return nil
}

func (m *LlamaServerManager) Restart() error {
	m.Lock()
	proc := m.cmd
	m.Unlock()

	if proc != nil && proc.Process != nil {
		fmt.Println("[LLAMA-SERVER] Restarting to pick up newly downloaded model...")
		_ = proc.Process.Kill()
		_, _ = proc.Process.Wait() // reap, avoid zombie
	}

	return m.Start()
}

// SetArgs replaces the argument list used for the next Start/Restart call.
// The base args (port, models-dir, preset) are passed in; this method slots
// the --rpc value in or removes it depending on whether rpcAddrs is empty.
func (m *LlamaServerManager) SetArgs(baseArgs []string, rpcAddrs string) {
	m.Lock()
	defer m.Unlock()
	args := make([]string, len(baseArgs))
	copy(args, baseArgs)
	if rpcAddrs != "" {
		args = append(args, "--rpc", rpcAddrs)
	}
	m.args = args
}

// RestartWithReason kills the running process and starts a fresh one using
// the current m.args. The reason string is included in the log line.
func (m *LlamaServerManager) RestartWithReason(reason string) error {
	m.Lock()
	proc := m.cmd
	m.Unlock()

	if proc != nil && proc.Process != nil {
		fmt.Printf("[LLAMA-SERVER] Restarting: %s\n", reason)
		_ = proc.Process.Kill()
		_, _ = proc.Process.Wait()
	}
	return m.Start()
}

func isZeroKeepAlive(v interface{}) bool {
	switch val := v.(type) {
	case float64:
		return val == 0
	case string:
		return val == "0" || val == "0s" || val == "0m"
	default:
		return false
	}
}

func waitForRouterReady(baseURL string, timeout time.Duration) {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		resp, err := rpcHTTPClient.Get(baseURL + "/models")
		if err == nil {
			resp.Body.Close()
			return
		}
		time.Sleep(200 * time.Millisecond)
	}
}

// downloadModel runs the Python sidecar to download a model from Hugging Face,
// reporting progress via onProgress as it goes, and cleaning up any stray
// .incomplete blob if ctx is cancelled or the download fails. It has no
// knowledge of HTTP - both the /ollama/api/pull handler and the standalone
// "pull" CLI command call this and handle PullProgress their own way.
func downloadModel(ctx context.Context, repo string, fileName string, onProgress func(PullProgress)) error {
	downloadCmd := exec.Command("python3", "/app/hf_progress_download.py", repo, fileName)
	downloadCmd.Env = append(os.Environ(), "HF_HUB_DISABLE_XET=1")

	stdout, err := downloadCmd.StdoutPipe()
	if err != nil {
		return err
	}
	downloadCmd.Stderr = os.Stderr
	if err := downloadCmd.Start(); err != nil {
		return err
	}

	scanner := bufio.NewScanner(stdout)
	var lastTotal int64 = 0
	var realDigest string = ""
	sentAnyProgress := false

	cleanupIncompleteBlobs := func(digest string) {
		if digest == "" {
			return
		}
		blobHash := strings.TrimPrefix(digest, "sha256:")
		home, _ := os.UserHomeDir()
		pattern := filepath.Join(home, ".cache", "huggingface", "hub", "models--*", "blobs", blobHash+".*.incomplete")
		matches, _ := filepath.Glob(pattern)
		for _, m := range matches {
			if rmErr := os.Remove(m); rmErr == nil {
				fmt.Printf("[CLEANUP] removed stale incomplete blob: %s\n", m)
			}
		}
	}

	lineCh := make(chan string)
	go func() {
		for scanner.Scan() {
			lineCh <- scanner.Text()
		}
		close(lineCh)
	}()

loop:
	for {
		select {
		case <-ctx.Done():
			_ = downloadCmd.Process.Kill()
			_, _ = downloadCmd.Process.Wait()
			cleanupIncompleteBlobs(realDigest)
			return ctx.Err()

		case line, chOk := <-lineCh:
			if !chOk {
				break loop
			}
			line = strings.TrimSpace(line)
			if line == "" {
				continue
			}

			var prog struct {
				Status    string `json:"status"`
				Digest    string `json:"digest"`
				Completed int64  `json:"completed"`
				Total     int64  `json:"total"`
				Done      bool   `json:"done"`
				Path      string `json:"path"`
			}
			if err := json.Unmarshal([]byte(line), &prog); err != nil {
				continue
			}

			if prog.Digest != "" {
				realDigest = prog.Digest
			}

			if prog.Status == "pulling manifest" {
				onProgress(PullProgress{Status: "pulling manifest"})
				continue
			}

			if prog.Done {
				break loop
			}

			if prog.Total > 0 {
				lastTotal = prog.Total
			}
			if lastTotal == 0 {
				continue
			}
			if prog.Completed == 0 && !sentAnyProgress {
				continue
			}
			sentAnyProgress = true

			onProgress(PullProgress{
				Status:    "pulling",
				Digest:    realDigest,
				Completed: prog.Completed,
				Total:     lastTotal,
			})
		}
	}

	if err := downloadCmd.Wait(); err != nil {
		cleanupIncompleteBlobs(realDigest)
		onProgress(PullProgress{Error: err, Done: true})
		return err
	}

	onProgress(PullProgress{Status: "success", Done: true})
	return nil
}

var rpcHTTPClient = &http.Client{
	Timeout: 10 * time.Minute, // Tarpeeksi pitkä aika raskaalle RPC-ajolle
	Transport: &http.Transport{
		ResponseHeaderTimeout: 5 * time.Minute,
		IdleConnTimeout:       30 * time.Second,
	},
}

func resolveRealModelFile(requestedModel string) string {
	clean := strings.TrimSuffix(requestedModel, ".gguf")
	return strings.TrimSuffix(clean, ":latest")
}

// resolveModelPath finds the real GGUF file path for a given repo:tag model
// identifier by reusing ScanHFCacheModels, so CLI inference and server mode
// agree on how models are located in the cache.
func resolveModelPath(modelName string, cacheDir string) (string, error) {
	models := ScanHFCacheModels(cacheDir)
	cleanName := resolveRealModelFile(modelName)
	for _, m := range models {
		if m.Name == cleanName {
			return m.RealPath, nil
		}
	}
	return "", fmt.Errorf("model %q not found in cache", modelName)
}

func ScanHFCacheModels(cacheDir string) []OllamaModelInfo {
	var foundModels []OllamaModelInfo
	repos, err := os.ReadDir(cacheDir)
	if err != nil {
		return foundModels
	}

	for _, repo := range repos {
		if repo.IsDir() && strings.HasPrefix(repo.Name(), "models--") {
			repoParts := strings.Split(strings.TrimPrefix(repo.Name(), "models--"), "--")
			if len(repoParts) < 2 {
				continue
			}
			repoID := strings.Join(repoParts, "/")

			refsPath := filepath.Join(cacheDir, repo.Name(), "refs", "main")
			commitHashBytes, err := os.ReadFile(refsPath)
			if err != nil {
				continue
			}
			commitHash := strings.TrimSpace(string(commitHashBytes))

			snapshotDir := filepath.Join(cacheDir, repo.Name(), "snapshots", commitHash)
			files, err := os.ReadDir(snapshotDir)
			if err != nil {
				continue
			}

			for _, f := range files {
				if !f.IsDir() && strings.HasSuffix(strings.ToLower(f.Name()), ".gguf") {
					fileName := f.Name()
					info, _ := f.Info()

					var tag string
					upperName := strings.ToUpper(fileName)
					if strings.Contains(upperName, "Q4_K_M") {
						tag = "Q4_K_M"
					} else if strings.Contains(upperName, "Q8_0") {
						tag = "Q8_0"
					} else if strings.Contains(upperName, "Q5_K_M") {
						tag = "Q5_K_M"
					} else if strings.Contains(upperName, "Q4_0") {
						tag = "Q4_0"
					} else {
						tag = strings.TrimSuffix(fileName, ".gguf")
					}

					fullIdentifier1 := fmt.Sprintf("%s:%s", repoID, tag)
					realPath := filepath.Join(snapshotDir, fileName)
					foundModels = append(foundModels, OllamaModelInfo{
						Name: fullIdentifier1, Model: fullIdentifier1, ModifiedAt: info.ModTime(), Size: info.Size(),
						Digest: fmt.Sprintf("sha256-%d", info.Size()), Details: map[string]interface{}{"format": "gguf", "family": "llama"},
						RealPath: realPath,
					})
				}
			}
		}
	}
	return foundModels
}

func getBinaryPath(name string) string {
	path, err := exec.LookPath(name)
	if err == nil {
		return path
	}
	localPaths := []string{fmt.Sprintf("./llama.cpp/build/bin/%s", name), fmt.Sprintf("./llama.cpp/%s", name)}
	for _, p := range localPaths {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	os.Exit(1)
	return ""
}

func runServe(port int, llamaport int, verbose bool, rpcAddrs string, discovery bool, discoveryPort int, discoveryWait time.Duration, promptArgs []string) {
	llamaBaseURL := fmt.Sprintf("http://localhost:%d", llamaport)

	home, _ := os.UserHomeDir()
	hfModelsDir := filepath.Join(home, ".cache", "huggingface", "hub")
	_ = os.MkdirAll(hfModelsDir, 0755)

	// Startup check: warn if no HF token is configured, since unauthenticated
	// requests are slower and gated/private repos will fail outright without one.
	if token := os.Getenv("HF_TOKEN"); token == "" {
		fmt.Println("[WARN] HF_TOKEN is not set - downloads will be unauthenticated (lower rate limits, gated/private repos will fail)")
	} else {
		masked := token
		if len(masked) > 8 {
			masked = masked[:4] + strings.Repeat("*", len(masked)-8) + masked[len(masked)-4:]
		}
		fmt.Printf("[DEBUG] HF_TOKEN is set (%s)\n", masked)
	}

	ctxTracker := NewCtxSizeTracker()
	presetMgr := NewModelPresetManager(filepath.Join(home, ".cache", "huggingface", "router.preset.ini"))

	if err := presetMgr.EnsureGlobalDefault(4096); err != nil {
		fmt.Printf("[PRESET] Failed to write global default: %v\n", err)
	}

	for modelID, ctxSize := range presetMgr.LoadCtxSizes() {
		ctxTracker.Record(modelID, ctxSize)
		fmt.Printf("[CTX] Seeded from preset file: %s = %d\n", modelID, ctxSize)
	}

	serverBin := getBinaryPath("llama-server")
	// baseArgs holds everything except --rpc, which is managed dynamically by
	// the discovery watcher (or set once here in manual/no-discovery mode).
	baseArgs := []string{"--port", strconv.Itoa(llamaport), "--models-dir", hfModelsDir, "--models-preset", presetMgr.path}
	baseArgs = append(baseArgs, promptArgs...)

	llamaServer := &LlamaServerManager{serverBin: serverBin}
	defer func() {
		llamaServer.Lock()
		if llamaServer.cmd != nil && llamaServer.cmd.Process != nil {
			llamaServer.cmd.Process.Kill()
		}
		llamaServer.Unlock()
	}()

	// --- RPC worker resolution ---
	// Manual mode: --rpc takes precedence over discovery; start immediately.
	// Discovery mode: hand off to startDiscoveryWatcher which handles the
	//   initial scan, the first llama-server Start(), and all future restarts.
	// Neither: start single-node immediately.
	if rpcAddrs != "" {
		fmt.Printf("[SERVE] Manual RPC workers: %s\n", rpcAddrs)
		llamaServer.SetArgs(baseArgs, rpcAddrs)
		if err := llamaServer.Start(); err != nil {
			fmt.Println("[ERROR] Failed to start llama-server:", err)
			os.Exit(1)
		}
	} else if discovery {
		// startDiscoveryWatcher blocks for initialWait, then starts llama-server
		// and watches for roster changes in the background. It does NOT return.
		// We run it in a goroutine so http.ListenAndServe can start below.
		waitReady := func() { waitForRouterReady(llamaBaseURL, 10*time.Second) }
		go startDiscoveryWatcher(discoveryWait, discoveryPort, baseArgs, llamaServer, waitReady)
		// Block until llama-server is up before we begin serving HTTP traffic.
		waitForRouterReady(llamaBaseURL, discoveryWait+15*time.Second)
	} else {
		llamaServer.SetArgs(baseArgs, "")
		if err := llamaServer.Start(); err != nil {
			fmt.Println("[ERROR] Failed to start llama-server:", err)
			os.Exit(1)
		}
	}

	http.HandleFunc("/ollama/api/pull", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			return
		}
		var pullReq OllamaPullRequest
		_ = json.NewDecoder(r.Body).Decode(&pullReq)
		w.Header().Set("Content-Type", "application/x-ndjson")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		flusher, ok := w.(http.Flusher)

		rawModel := strings.TrimSuffix(pullReq.Model, ":latest")
		parts := strings.Split(rawModel, ":")
		if len(parts) < 2 {
			return
		}
		repo, fileName := parts[0], parts[1]
		if !strings.HasSuffix(strings.ToLower(fileName), ".gguf") {
			fileName += ".gguf"
		}

		writeLine := func(v interface{}) {
			out, _ := json.Marshal(v)
			_, _ = w.Write(out)
			_, _ = w.Write([]byte("\n"))
			if ok {
				flusher.Flush()
			}
		}

		err := downloadModel(r.Context(), repo, fileName, func(p PullProgress) {
			if p.Status == "pulling manifest" {
				writeLine(map[string]interface{}{"status": "pulling manifest"})
				return
			}
			if p.Error != nil {
				writeLine(map[string]interface{}{"status": "error", "error": p.Error.Error()})
				return
			}
			if p.Done {
				writeLine(map[string]interface{}{"status": "success", "done": true})
				return
			}

			digestForStatus := p.Digest
			if digestForStatus == "" {
				digestForStatus = "sha256:" + strings.Repeat("0", 64)
			}
			statusDigest := strings.TrimPrefix(digestForStatus, "sha256:")
			if len(statusDigest) > 12 {
				statusDigest = statusDigest[:12]
			}
			writeLine(map[string]interface{}{
				"status":    fmt.Sprintf("pulling %s", statusDigest),
				"digest":    digestForStatus,
				"completed": p.Completed,
				"total":     p.Total,
			})
		})

		if err != nil {
			return // error already written via onProgress before downloadModel returned
		}

		// llama-server only scans --models-dir/--models-preset once at its own
		// startup, so restart it to pick up the newly downloaded model.
		if err := llamaServer.Restart(); err != nil {
			fmt.Println("[ERROR] Failed to restart llama-server after pull:", err)
		}
	})

	http.HandleFunc("/ollama/api/chat", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			return
		}
		var chatReq struct {
			Model     string                 `json:"model"`
			Messages  []map[string]string    `json:"messages"`
			Stream    bool                   `json:"stream"`
			Options   map[string]interface{} `json:"options"`
			KeepAlive interface{}            `json:"keep_alive"`
		}
		bodyBytes, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(bodyBytes, &chatReq)

		// Open WebUI's eject button signals "unload this model" by POSTing here
		// with an empty prompt and keep_alive: 0 - mirror real Ollama's behavior
		// by forwarding it to llama-server's router /models/unload instead of
		// treating it as a real completion request.
		if len(chatReq.Messages) == 0 && isZeroKeepAlive(chatReq.KeepAlive) {
			fmt.Printf("[UNLOAD] Unloading model via router: %s\n", chatReq.Model)
			unloadBody, _ := json.Marshal(map[string]string{"model": chatReq.Model})
			resp, err := rpcHTTPClient.Post(llamaBaseURL+"/models/unload", "application/json", bytes.NewBuffer(unloadBody))
			if err != nil {
				fmt.Printf("[UNLOAD] error calling router unload: %v\n", err)
			} else {
				defer resp.Body.Close()
			}

			w.Header().Set("Content-Type", "application/json")
			respJSON, _ := json.Marshal(map[string]interface{}{
				"model":       chatReq.Model,
				"created_at":  time.Now().UTC(),
				"message":     map[string]string{"role": "assistant", "content": ""},
				"done_reason": "unload",
				"done":        true,
			})
			_, _ = w.Write(respJSON)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")

		var openAIMessages []OpenAIChatMessage
		for _, msg := range chatReq.Messages {
			openAIMessages = append(openAIMessages, OpenAIChatMessage{Role: msg["role"], Content: msg["content"]})
		}

		// Build the base request as a map so we can merge in arbitrary llama.cpp/Ollama sampler options
		if verbose {
			fmt.Printf("[DEBUG] Model to load: %s\n", resolveRealModelFile(chatReq.Model))
		}
		openAIReqMap := map[string]interface{}{
			"model":    resolveRealModelFile(chatReq.Model),
			"messages": openAIMessages,
			"stream":   chatReq.Stream,
		}
		if chatReq.Stream {
			openAIReqMap["stream_options"] = map[string]bool{"include_usage": true}
		}

		// Safety limit, if num_predict is missing
		//maxTokens := 4096

		// Route ALL Ollama options-fields to llama-server (names are mainly identical)
		if chatReq.Options != nil {
			for key, val := range chatReq.Options {
				switch key {
				case "num_predict":
					if floatVal, ok := val.(float64); ok && floatVal > 0 {
						//maxTokens = int(floatVal)
						openAIReqMap["max_tokens"] = int(floatVal)
					}
				case "num_ctx":
					if floatVal, ok := val.(float64); ok && floatVal > 0 {
						requestedCtx := int(floatVal)
						resolvedModel := resolveRealModelFile(chatReq.Model)

						if ctxTracker.NeedsUpdate(resolvedModel, requestedCtx) {
							fmt.Printf("[CTX] %s: changing ctx-size to %d\n", resolvedModel, requestedCtx)

							if err := presetMgr.SetCtxSize(resolvedModel, requestedCtx); err != nil {
								fmt.Printf("[CTX] failed to write preset for %s: %v\n", resolvedModel, err)
								// Don't update the tracker or trigger an unload if the write failed -
								// we'd otherwise think the change took effect when it didn't.
							} else {
								// The router only parses --models-preset once at its own startup,
								// so a per-model unload/reload alone won't pick up a freshly
								// edited preset file - the whole router process needs restarting.
								if err := llamaServer.Restart(); err != nil {
									fmt.Printf("[CTX] failed to restart llama-server after ctx-size change: %v\n", err)
								} else {
									// Give the router a moment to come back up before this same request
									// continues on to the actual chat completion call below.
									waitForRouterReady(llamaBaseURL, 4*time.Second)
								}
								ctxTracker.Record(resolvedModel, requestedCtx)
							}
						}
					}
					continue
				default:
					// temperature, top_k, top_p, min_p, repeat_penalty, repeat_last_n,
					// frequency_penalty, presence_penalty, seed, stop, mirostat, mirostat_eta,
					// mirostat_tau, typical_p jne. - names match with llama-server
					openAIReqMap[key] = val
				}
			}
		}
		//openAIReqMap["max_tokens"] = maxTokens
		if verbose {
			fmt.Printf("[DEBUG] options received: %+v\n", chatReq.Options)
		}

		backendJSON, _ := json.Marshal(openAIReqMap)

		// --- STATE 1: NON-STREAMED REQUEST (Follow-up questions) ---
		if !chatReq.Stream {
			resp, err := rpcHTTPClient.Post(llamaBaseURL+"/v1/chat/completions", "application/json", bytes.NewBuffer(backendJSON))
			if err != nil {
				fmt.Printf("Llama-server connection error in non-streamed request: %v\n", err)
				w.WriteHeader(http.StatusInternalServerError)
				errOut, _ := json.Marshal(map[string]interface{}{"error": err.Error()})
				_, _ = w.Write(errOut)
				return
			}
			defer resp.Body.Close()

			if resp.StatusCode != http.StatusOK {
				var errResp map[string]interface{}
				_ = json.NewDecoder(resp.Body).Decode(&errResp)
				fmt.Printf("Llama-server error in non-streamed request: %v", errResp)

				w.WriteHeader(http.StatusInternalServerError)
				errOut, _ := json.Marshal(map[string]interface{}{
					"error": fmt.Sprintf("llama-server error: %v", errResp),
				})
				_, _ = w.Write(errOut)
				return
			}

			// ... existing success-path decoding/response-building continues unchanged ...

			var openAIResp struct {
				Choices []struct {
					Message struct {
						Content          string `json:"content"`
						ReasoningContent string `json:"reasoning_content"`
					} `json:"message"`
				} `json:"choices"`
				Usage   LlamaUsage   `json:"usage"`
				Timings LlamaTimings `json:"timings"`
			}
			_ = json.NewDecoder(resp.Body).Decode(&openAIResp)

			messageMap := map[string]string{"role": "assistant", "content": ""}
			if len(openAIResp.Choices) > 0 {
				if openAIResp.Choices[0].Message.ReasoningContent != "" {
					messageMap["thinking"] = openAIResp.Choices[0].Message.ReasoningContent
				}
				messageMap["content"] = openAIResp.Choices[0].Message.Content
			}

			promptNs := int64(openAIResp.Timings.PromptMS * 1e6)
			evalNs := int64(openAIResp.Timings.PredictedMS * 1e6)

			outJSON, _ := json.Marshal(map[string]interface{}{
				"model":                chatReq.Model,
				"created_at":           time.Now().UTC(),
				"message":              messageMap,
				"done":                 true,
				"total_duration":       promptNs + evalNs,
				"prompt_eval_count":    openAIResp.Usage.PromptTokens,
				"prompt_eval_duration": promptNs,
				"eval_count":           openAIResp.Usage.CompletionTokens,
				"eval_duration":        evalNs,
			})
			_, _ = w.Write(outJSON)
			return
		}

		// --- STATE 2: STREAMED REQUEST (Normal chat) ---
		flusher, ok := w.(http.Flusher)
		initChunk, _ := json.Marshal(map[string]interface{}{
			"model":      chatReq.Model,
			"created_at": time.Now().UTC(),
			"message":    map[string]string{"role": "assistant", "content": ""},
			"done":       false,
		})
		_, _ = w.Write(initChunk)
		_, _ = w.Write([]byte("\n"))
		if ok {
			flusher.Flush()
		}

		req, _ := http.NewRequest("POST", llamaBaseURL+"/v1/chat/completions", bytes.NewBuffer(backendJSON))
		req.Header.Set("Content-Type", "application/json")
		resp, err := rpcHTTPClient.Do(req)
		if err != nil {
			fmt.Printf("Llama-server connection error in streamed request: %v\n", err)
			errOut, _ := json.Marshal(map[string]interface{}{
				"model":      chatReq.Model,
				"created_at": time.Now().UTC(),
				"done":       true,
				"error":      err.Error(),
			})
			_, _ = w.Write(errOut)
			_, _ = w.Write([]byte("\n"))
			if ok {
				flusher.Flush()
			}
			return
		}
		defer resp.Body.Close()

		var lastUsage LlamaUsage
		var lastTimings LlamaTimings

		// Make sure always free chat to Open WebUI, even if llama-server crashes or returns error. This is to avoid the client hanging forever without a final "done" message.
		defer func() {
			promptNs := int64(lastTimings.PromptMS * 1e6)
			evalNs := int64(lastTimings.PredictedMS * 1e6)
			finalJSON, _ := json.Marshal(map[string]interface{}{
				"model":                chatReq.Model,
				"created_at":           time.Now().UTC(),
				"done":                 true,
				"total_duration":       promptNs + evalNs,
				"prompt_eval_count":    lastUsage.PromptTokens,
				"prompt_eval_duration": promptNs,
				"eval_count":           lastUsage.CompletionTokens,
				"eval_duration":        evalNs,
			})
			_, _ = w.Write(finalJSON)
			_, _ = w.Write([]byte("\n"))
			if ok {
				flusher.Flush()
			}
		}()

		// If server returns an error immediately (e.g., too many tokens requested), we need to read the error response and return it to the client.
		if resp.StatusCode != http.StatusOK {
			var errResp map[string]interface{}
			_ = json.NewDecoder(resp.Body).Decode(&errResp)
			fmt.Printf("Llama-server error in streamed request: %v", errResp)

			errOut, _ := json.Marshal(map[string]interface{}{
				"model":      chatReq.Model,
				"created_at": time.Now().UTC(),
				"done":       true,
				"error":      fmt.Sprintf("llama-server error: %v", errResp),
			})
			_, _ = w.Write(errOut)
			_, _ = w.Write([]byte("\n"))
			if ok {
				flusher.Flush()
			}
			return
		}

		reader := bufio.NewReader(resp.Body)
		for {
			line, err := reader.ReadString('\n')
			if err == io.EOF {
				break
			}
			if err != nil {
				return
			}
			line = strings.TrimSpace(line)
			if !strings.HasPrefix(line, "data:") {
				continue
			}
			dataStr := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
			if dataStr == "[DONE]" {
				break
			}

			var chunk OpenAIChatChunk
			if err := json.Unmarshal([]byte(dataStr), &chunk); err != nil {
				continue
			}

			// Get usage/timings always when they are present (usually in the last, choices-empty chunk)
			if chunk.Usage.CompletionTokens > 0 || chunk.Usage.PromptTokens > 0 {
				lastUsage = chunk.Usage
			}
			if chunk.Timings.PredictedMS > 0 || chunk.Timings.PromptMS > 0 {
				lastTimings = chunk.Timings
			}

			if len(chunk.Choices) == 0 {
				continue
			} // usage-only chunk, not containing delta text

			if chunk.Choices[0].FinishReason != "" && chunk.Choices[0].FinishReason != "null" {
				continue // Do not break here - usage/timings chunk may still come
			}

			contentWord := chunk.Choices[0].Delta.Content
			thinkingWord := chunk.Choices[0].Delta.ReasoningContent
			if contentWord == "" && thinkingWord == "" {
				continue
			}

			messageMap := map[string]string{"role": "assistant"}
			if thinkingWord != "" {
				messageMap["thinking"] = thinkingWord
			} else if contentWord != "" {
				messageMap["content"] = contentWord
			}

			outJSON, _ := json.Marshal(map[string]interface{}{
				"model":      chatReq.Model,
				"created_at": time.Now().UTC(),
				"message":    messageMap,
				"done":       false,
			})
			_, _ = w.Write(outJSON)
			_, _ = w.Write([]byte("\n"))
			if ok {
				flusher.Flush()
			}
		}

		promptNs := int64(lastTimings.PromptMS * 1e6)
		evalNs := int64(lastTimings.PredictedMS * 1e6)
		finalJSON, _ := json.Marshal(map[string]interface{}{
			"model":                chatReq.Model,
			"created_at":           time.Now().UTC(),
			"done":                 true,
			"total_duration":       promptNs + evalNs,
			"prompt_eval_count":    lastUsage.PromptTokens,
			"prompt_eval_duration": promptNs,
			"eval_count":           lastUsage.CompletionTokens,
			"eval_duration":        evalNs,
		})
		_, _ = w.Write(finalJSON)
		_, _ = w.Write([]byte("\n"))
		if ok {
			flusher.Flush()
		}
	})

	http.HandleFunc("/ollama/api/generate", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			return
		}

		bodyBytes, err := io.ReadAll(r.Body)
		if err != nil {
			return
		}

		var genReq struct {
			Model     string      `json:"model"`
			Prompt    string      `json:"prompt"`
			KeepAlive interface{} `json:"keep_alive"`
		}
		_ = json.Unmarshal(bodyBytes, &genReq)

		// Open WebUI's eject button signals "unload this model" by POSTing here
		// with an empty prompt and keep_alive: 0 - mirror real Ollama's behavior
		// by forwarding it to llama-server's router /models/unload instead of
		// treating it as a real completion request.
		if genReq.Prompt == "" && isZeroKeepAlive(genReq.KeepAlive) {
			fmt.Printf("[UNLOAD] /ollama/api/generate unload signal for model: %s\n", genReq.Model)

			unloadBody, _ := json.Marshal(map[string]string{"model": genReq.Model})
			resp, err := rpcHTTPClient.Post(llamaBaseURL+"/models/unload", "application/json", bytes.NewBuffer(unloadBody))
			w.Header().Set("Content-Type", "application/json")
			if err != nil {
				fmt.Printf("[UNLOAD] error forwarding to llama-server router: %v\n", err)
				errResp, _ := json.Marshal(map[string]interface{}{"error": err.Error()})
				_, _ = w.Write(errResp)
				return
			}
			defer resp.Body.Close()

			// Ollama's real response to a keep_alive:0 unload is a single JSON
			// object with done_reason "unload" and done: true, no response text.
			doneResp, _ := json.Marshal(map[string]interface{}{
				"model":       genReq.Model,
				"created_at":  time.Now().UTC(),
				"response":    "",
				"done":        true,
				"done_reason": "unload",
			})
			_, _ = w.Write(doneResp)
			return
		}

		// Not an unload signal - proceed with the existing reverse-proxy behavior
		// for real generation requests.
		r.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))
		originUrl, _ := url.Parse(llamaBaseURL)
		r.URL.Path = "/completion"
		r.Host = originUrl.Host
		httputil.NewSingleHostReverseProxy(originUrl).ServeHTTP(w, r)
	})

	http.HandleFunc("/openai/v1/models", func(w http.ResponseWriter, r *http.Request) {
		models := ScanHFCacheModels(hfModelsDir)
		var openAIModels []OpenAIModelPrice
		for _, m := range models {
			openAIModels = append(openAIModels, OpenAIModelPrice{ID: m.Name, Object: "model", Created: time.Now().Unix(), OwnedBy: "LlamaNexus"})
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(OpenAIModelsResponse{Object: "list", Data: openAIModels})
	})
	http.HandleFunc("/openai/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		originUrl, _ := url.Parse(llamaBaseURL)
		r.URL.Path = "/v1/chat/completions"
		r.Host = originUrl.Host
		httputil.NewSingleHostReverseProxy(originUrl).ServeHTTP(w, r)
	})

	http.HandleFunc("/ollama/api/version", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]string{"version": "0.1.48"})
	})

	http.HandleFunc("/ollama/api/ps", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")

		resp, err := rpcHTTPClient.Get(llamaBaseURL + "/models")
		if err != nil {
			// llama-server router unreachable - report no running models rather than erroring,
			// matching Ollama's behavior of an empty list when nothing is loaded.
			_ = json.NewEncoder(w).Encode(OllamaPsResponse{Models: []map[string]interface{}{}})
			return
		}
		defer resp.Body.Close()

		var routerResp RouterModelsResponse
		if err := json.NewDecoder(resp.Body).Decode(&routerResp); err != nil {
			_ = json.NewEncoder(w).Encode(OllamaPsResponse{Models: []map[string]interface{}{}})
			return
		}

		var runningModels []map[string]interface{}
		for _, m := range routerResp.Data {
			if m.Status.Value != "loaded" {
				continue
			}

			var sizeBytes int64 = 0
			if m.Meta != nil {
				sizeBytes = m.Meta.Size
			}

			runningModels = append(runningModels, map[string]interface{}{
				"name":   m.ID,
				"model":  m.ID,
				"size":   sizeBytes,
				"digest": fmt.Sprintf("sha256-%d", sizeBytes), // same placeholder scheme used elsewhere in this file
				"details": map[string]interface{}{
					"parent_model":       "",
					"format":             "gguf",
					"family":             "llama",
					"families":           []string{"llama"},
					"parameter_size":     "",
					"quantization_level": "",
				},
				// Router mode doesn't expose a keep_alive/expiry timer the way Ollama
				// does (LRU eviction is automatic, not time-based), so report a
				// far-future expiry rather than inventing a real countdown.
				"expires_at": time.Now().Add(24 * time.Hour).UTC().Format(time.RFC3339),
				"size_vram":  sizeBytes,
			})
		}

		if runningModels == nil {
			runningModels = []map[string]interface{}{}
		}

		_ = json.NewEncoder(w).Encode(map[string]interface{}{"models": runningModels})
	})

	http.HandleFunc("/ollama/api/tags", func(w http.ResponseWriter, r *http.Request) {
		models := ScanHFCacheModels(hfModelsDir)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(OllamaTagsResponse{Models: models})
	})

	fmt.Printf("[SERVE] LlamaNexus Unified Reasoning Proxy portissa %d...\n", port)
	_ = http.ListenAndServe(fmt.Sprintf(":%d", port), nil)
}

func runPull(modelName string, verbose bool) {
	rawModel := strings.TrimSuffix(modelName, ":latest")
	parts := strings.Split(rawModel, ":")
	if len(parts) < 2 {
		fmt.Println("[ERROR] model must be specified as repo:tag, e.g. Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M")
		os.Exit(1)
	}
	repo, fileName := parts[0], parts[1]
	if !strings.HasSuffix(strings.ToLower(fileName), ".gguf") {
		fileName += ".gguf"
	}

	fmt.Printf("[PULL] Downloading %s...\n", modelName)

	err := downloadModel(context.Background(), repo, fileName, func(p PullProgress) {
		switch {
		case p.Error != nil:
			fmt.Println("[PULL] error:", p.Error)
		case p.Status == "pulling manifest":
			fmt.Println("[PULL] resolving manifest...")
		case p.Done:
			fmt.Println("[PULL] done.")
		case p.Total > 0:
			pct := float64(p.Completed) / float64(p.Total) * 100
			fmt.Printf("\r[PULL] %.1f%% (%d / %d bytes)", pct, p.Completed, p.Total)
		}
	})

	fmt.Println()
	if err != nil {
		fmt.Println("[ERROR] pull failed:", err)
		os.Exit(1)
	}
}

// resolveModelPathByRepo is a fallback for when the user's original model
// string (e.g. an exact filename) doesn't match ScanHFCacheModels' inferred
// short-tag identifier. It matches on repo name alone and returns the most
// recently modified file for that repo - a reasonable choice immediately
// after a fresh download, since that's the file that was just written.
func resolveModelPathByRepo(modelName string, cacheDir string) (string, error) {
	repo := strings.SplitN(resolveRealModelFile(modelName), ":", 2)[0]
	models := ScanHFCacheModels(cacheDir)

	var best OllamaModelInfo
	found := false
	for _, m := range models {
		modelRepo := strings.SplitN(m.Name, ":", 2)[0]
		if modelRepo == repo && (!found || m.ModifiedAt.After(best.ModifiedAt)) {
			best = m
			found = true
		}
	}
	if !found {
		return "", fmt.Errorf("no cached model found for repo %q", repo)
	}
	return best.RealPath, nil
}

func runCliInference(model string, verbose bool, promptArgs []string) {
	home, err := os.UserHomeDir()
	if err != nil {
		fmt.Println("[ERROR] could not determine home directory:", err)
		os.Exit(1)
	}
	hfCacheDir := filepath.Join(home, ".cache", "huggingface", "hub")

	modelPath, err := resolveModelPath(model, hfCacheDir)
	if err != nil {
		fmt.Printf("[RUN] model %s not found in cache, downloading...\n", model)
		runPull(model, verbose)

		// After downloading, re-scan rather than trust the original input string -
		// the user may have typed a full filename or a short quant tag, and
		// ScanHFCacheModels always normalizes to its own short-tag scheme.
		// Match on repo alone if the exact string still doesn't resolve.
		modelPath, err = resolveModelPath(model, hfCacheDir)
		if err != nil {
			modelPath, err = resolveModelPathByRepo(model, hfCacheDir)
		}
		if err != nil {
			fmt.Println("[ERROR] model still not found after pull:", err)
			os.Exit(1)
		}
	}

	if verbose {
		fmt.Printf("[DEBUG] Resolved model %s to path %s\n", model, modelPath)
	}

	cliBin := getBinaryPath("llama-cli")
	args := []string{"-m", modelPath, "-st"}
	args = append(args, promptArgs...)

	cmd := exec.Command(cliBin, args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		fmt.Println("[ERROR] llama-cli exited with error:", err)
		os.Exit(1)
	}
}

func runRpcServer(port int, discovery bool, discoveryPort int, advertiseAddr string, promptArgs []string) {
	rpcBin := getBinaryPath("rpc-server")
	args := []string{"-H", "0.0.0.0", "-p", strconv.Itoa(port)}
	args = append(args, promptArgs...)
	cmd := exec.Command(rpcBin, args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	// Start the heartbeat broadcaster before launching rpc-server so that
	// serve nodes scanning at startup can already see this worker.
	var stopHeartbeat chan struct{}
	if discovery {
		// Resolve the address to advertise in heartbeats.
		// Priority: --advertise-addr flag > LLAMANEXUS_ADVERTISE_ADDR env var > auto-detect.
		// Auto-detection routes toward 255.255.255.255 to prefer the LAN interface
		// over Docker bridge IPs, but an explicit override is always safer in
		// complex network topologies (multi-NIC, VLAN, macvlan, host-network, etc.).
		rpcAddr := advertiseAddr
		if rpcAddr == "" {
			rpcAddr = os.Getenv("LLAMANEXUS_ADVERTISE_ADDR")
		}
		if rpcAddr == "" {
			rpcAddr = outboundIP()
		}
		// If the value has no port (e.g. "192.168.0.120"), append the RPC port.
		if !strings.Contains(rpcAddr, ":") {
			rpcAddr = fmt.Sprintf("%s:%d", rpcAddr, port)
		}
		fmt.Printf("[WORKER] Advertising RPC address: %s\n", rpcAddr)
		stopHeartbeat = make(chan struct{})
		go broadcastHeartbeat(rpcAddr, discoveryPort, stopHeartbeat)
	}

	fmt.Printf("[WORKER] RPC-server active: port=%d discovery=%t...\n", port, discovery)
	if err := cmd.Run(); err != nil {
		fmt.Printf("[WORKER] rpc-server exited: %v\n", err)
	}

	// Stop heartbeat after the rpc-server process exits.
	if stopHeartbeat != nil {
		close(stopHeartbeat)
	}
}
