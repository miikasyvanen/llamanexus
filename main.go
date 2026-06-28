package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
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

type OpenAIChatRequest struct {
	Model         string              `json:"model"`
	Messages      []OpenAIChatMessage `json:"messages"`
	Temperature   float64             `json:"temperature,omitempty"`
	MaxTokens     int                 `json:"max_tokens,omitempty"`
	Stream        bool                `json:"stream"`
	StreamOptions *StreamOptions      `json:"stream_options,omitempty"`
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

type llamaTimings struct {
	PromptMS    float64 `json:"prompt_ms"`
	PredictedMS float64 `json:"predicted_ms"`
}
type llamaUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
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

// llamaBaseURL is the single source of truth for llama-server's address.
// Computed once from the --llamaport flag at startup and reused by every
// handler that talks to llama-server, so changing the port only requires
// changing this one value.
//var llamaBaseURL string

var rpcHTTPClient = &http.Client{
	Timeout: 10 * time.Minute, // Tarpeeksi pitkä aika raskaalle RPC-ajolle
	Transport: &http.Transport{
		ResponseHeaderTimeout: 5 * time.Minute,
		IdleConnTimeout:       30 * time.Second,
	},
}

func main() {
	portFlag := flag.IntP("port", "", 0, "Määritä API/RPC portti")
	llamaPortFlag := flag.IntP("llamaport", "", 8080, "Määritä Llama.cpp API portti")
	modelFlag := flag.StringP("model", "m", DefaultModel, "Käytettävä GGUF-malli")
	verboseFlag := flag.BoolP("verbose", "v", false, "Verbose-tila")
	//ctxFlag := flag.IntP("ctx-size", "c", 4096, "Kontekstin koko")

	flag.Parse()

	fmt.Println("proxy port:", *portFlag)
	fmt.Println("llama port:", *llamaPortFlag)

	port := *portFlag
	llamaport := *llamaPortFlag
	verbose := *verboseFlag
	model := *modelFlag
	//ctx := *ctxFlag

	args := flag.Args()
	command := ""
	if len(args) > 0 {
		command = args[0]
	} else {
		os.Exit(1)
	}
	fmt.Printf("Command line args:\n")
	for i := range args {
		fmt.Printf("arg %d: %s\n", i, args[i])
	}

	if port == 0 {
		if command == "worker" {
			port = 50052
		} else {
			port = 11434
		}
	}

	if command == "serve" {
		fmt.Printf("[SERVE] Käynnistetään llama-server: port=%d llama port=%d verbose=%t\n", port, llamaport, verbose)
		runServe(port, llamaport, verbose, args[1:])
	} else if command == "run" {
		fmt.Printf("[RUN] Käynnistetään llama-cli: model=%s verbose=%t\n", model, verbose)
		runCliInference(model, verbose, args[1:])
	} else if command == "worker" {
		fmt.Printf("[WORKER] Käynnistetään worker: port=%d\n", port)
		runRpcServer(port)
	} else {
		flag.Usage()
		os.Exit(1)
	}
}

func resolveRealModelFile(requestedModel string) string {
	clean := strings.TrimSuffix(requestedModel, ".gguf")
	return strings.TrimSuffix(clean, ":latest")
}

func findModelInHFCache(modelName string, cacheDir string) string {
	var foundPath string
	_ = filepath.Walk(cacheDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil
		}
		if info.Mode()&os.ModeSymlink != 0 {
			realPath, err := os.Readlink(path)
			if err == nil {
				if !filepath.IsAbs(realPath) {
					realPath = filepath.Join(filepath.Dir(path), realPath)
				}
				realInfo, err := os.Stat(realPath)
				if err == nil && realInfo.Size() > 100*1024*1024 && strings.Contains(path, modelName) {
					foundPath = realPath
					return filepath.SkipDir
				}
			}
		}
		if !info.IsDir() && info.Size() > 100*1024*1024 {
			if strings.Contains(path, "blobs") {
				foundPath = path
				return filepath.SkipDir
			}
		}
		return nil
	})
	return foundPath
}

func ensureModel(modelName string, verbose bool) string {
	home, err := os.UserHomeDir()
	if err != nil {
		os.Exit(1)
	}
	hfCacheDir := filepath.Join(home, ".cache", "huggingface", "hub")
	_ = os.MkdirAll(hfCacheDir, 0755)
	return findModelInHFCache(modelName, hfCacheDir)
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
					foundModels = append(foundModels, OllamaModelInfo{
						Name: fullIdentifier1, Model: fullIdentifier1, ModifiedAt: info.ModTime(), Size: info.Size(),
						Digest: fmt.Sprintf("sha256-%d", info.Size()), Details: map[string]interface{}{"format": "gguf", "family": "llama"},
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

func runServe(port int, llamaport int, verbose bool, promptArgs []string) {
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
	args := []string{"--port", strconv.Itoa(llamaport), "--models-dir", hfModelsDir, "--models-preset", presetMgr.path}
	args = append(args, promptArgs...)

	llamaServer := &LlamaServerManager{serverBin: serverBin, args: args}
	if err := llamaServer.Start(); err != nil {
		fmt.Println("[ERROR] Failed to start llama-server:", err)
		os.Exit(1)
	}
	defer func() {
		llamaServer.Lock()
		if llamaServer.cmd != nil && llamaServer.cmd.Process != nil {
			llamaServer.cmd.Process.Kill()
		}
		llamaServer.Unlock()
	}()

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

		rawModel := pullReq.Model
		rawModel = strings.TrimSuffix(rawModel, ":latest")
		parts := strings.Split(rawModel, ":")
		if len(parts) < 2 {
			return
		}
		repo, fileName := parts[0], parts[1]
		if !strings.HasSuffix(strings.ToLower(fileName), ".gguf") {
			fileName += ".gguf"
		}

		downloadCmd := exec.Command("python3", "/app/hf_progress_download.py", repo, fileName)
		downloadCmd.Env = append(os.Environ(), "HF_HUB_DISABLE_XET=1")

		stdout, err := downloadCmd.StdoutPipe()
		if err != nil {
			return
		}
		downloadCmd.Stderr = os.Stderr
		if err := downloadCmd.Start(); err != nil {
			return
		}

		scanner := bufio.NewScanner(stdout)
		ctx := r.Context()

		var lastTotal int64 = 0
		var realDigest string = ""
		sentAnyProgress := false

		// cleanupIncompleteBlobs removes any stray .incomplete temp files for the
		// given digest, working around a known huggingface_hub bug where killed/
		// failed downloads leave randomly-named partial blobs that are never
		// resumed or cleaned up (huggingface/huggingface_hub#4196).
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

		// Run the scanner in its own goroutine so we can select between new lines
		// and context cancellation (bufio.Scanner.Scan() can't be interrupted directly).
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
				// Client disconnected or cancelled the pull - kill the subprocess
				// and clean up its partial blob before returning.
				_ = downloadCmd.Process.Kill()
				_, _ = downloadCmd.Process.Wait() // reap the process, avoid zombie
				cleanupIncompleteBlobs(realDigest)
				return

			case line, chOk := <-lineCh:
				if !chOk {
					break loop // scanner finished (EOF or read error)
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
					manifestResp, _ := json.Marshal(map[string]interface{}{
						"status": "pulling manifest",
					})
					_, _ = w.Write(manifestResp)
					_, _ = w.Write([]byte("\n"))
					if ok {
						flusher.Flush()
					}
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

				// Hold back the first visible update until we see real movement,
				// avoiding a 0%/100% flash before tqdm reports genuine progress.
				if prog.Completed == 0 && !sentAnyProgress {
					continue
				}
				sentAnyProgress = true

				digestForStatus := realDigest
				if digestForStatus == "" {
					digestForStatus = "sha256:" + strings.Repeat("0", 64)
				}
				statusDigest := strings.TrimPrefix(digestForStatus, "sha256:")
				if len(statusDigest) > 12 {
					statusDigest = statusDigest[:12]
				}

				progResp, _ := json.Marshal(map[string]interface{}{
					"status":    fmt.Sprintf("pulling %s", statusDigest),
					"digest":    digestForStatus,
					"completed": prog.Completed,
					"total":     lastTotal,
				})
				_, _ = w.Write(progResp)
				_, _ = w.Write([]byte("\n"))
				if ok {
					flusher.Flush()
				}
			}
		}

		if err := downloadCmd.Wait(); err != nil {
			// Download failed (not user-cancelled, since that returns earlier) -
			// clean up any partial blob left behind here too.
			cleanupIncompleteBlobs(realDigest)
			errResp, _ := json.Marshal(map[string]interface{}{"status": "error", "error": err.Error()})
			_, _ = w.Write(errResp)
			_, _ = w.Write([]byte("\n"))
			if ok {
				flusher.Flush()
			}
			return
		}

		successResp, _ := json.Marshal(map[string]interface{}{"status": "success", "done": true})
		_, _ = w.Write(successResp)
		_, _ = w.Write([]byte("\n"))
		if ok {
			flusher.Flush()
		}

		// llama-server only scans --models-dir once at its own startup, so restart
		// it to pick up the newly downloaded model without requiring a full
		// container restart.
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
				fmt.Printf("Llama-server connection error ei-streamissa: %v\n", err)
				w.WriteHeader(http.StatusInternalServerError)
				errOut, _ := json.Marshal(map[string]interface{}{"error": err.Error()})
				_, _ = w.Write(errOut)
				return
			}
			defer resp.Body.Close()

			if resp.StatusCode != http.StatusOK {
				var errResp map[string]interface{}
				_ = json.NewDecoder(resp.Body).Decode(&errResp)
				fmt.Printf("Llama-server palautti virheen ei-streamissa: %v", errResp)

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
			fmt.Printf("Llama-server connection error streamissa: %v\n", err)
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
		r.URL.Path = "/completion"
		r.Host = "localhost:8080"
		originUrl, _ := url.Parse(llamaBaseURL)
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
		r.URL.Path = "/v1/chat/completions"
		r.Host = "localhost:8080"
		originUrl, _ := url.Parse(llamaBaseURL)
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

func runCliInference(model string, verbose bool, promptArgs []string) {
	modelPath := ensureModel(model, verbose)
	cliBin := getBinaryPath("llama-cli")
	args := []string{"-m", modelPath}
	args = append(args, strings.Join(promptArgs, " "))
	cmd := exec.Command(cliBin, args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	_ = cmd.Run()
}

func runRpcServer(port int) {
	rpcBin := getBinaryPath("rpc-server")
	cmd := exec.Command(rpcBin, "-p", strconv.Itoa(port))
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	fmt.Printf("[WORKER] rpc-server aktiivinen portissa %d...\n", port)
	_ = cmd.Run()
}
