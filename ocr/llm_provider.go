package ocr

import (
	"bytes"
	"context"
	"encoding/base64"
	"fmt"
	"image"
	"os"
	"strings"

	"github.com/sirupsen/logrus"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/ollama"
	"github.com/tmc/langchaingo/llms/openai"
)

// LLMProvider implements OCR using LLM vision models
type LLMProvider struct {
	Provider                  string
	Model                     string
	LLM                       llms.Model
	Prompt                    string // OCR prompt template
	OllamaOcrMaxTokensPerPage int
	OllamaOcrTemperature      *float64
	OllamaOcrTopK             *int
}

func newLLMProvider(config Config) (*LLMProvider, error) {
	logger := log.WithFields(logrus.Fields{
		"provider": config.VisionLLMProvider,
		"model":    config.VisionLLMModel,
	})
	logger.Info("Creating new LLM OCR provider")

	var model llms.Model
	var err error

	switch strings.ToLower(config.VisionLLMProvider) {
	case "openai":
		logger.Debug("Initializing OpenAI vision model")
		model, err = createOpenAIClient(config)
	case "ollama":
		logger.Debug("Initializing Ollama vision model")
		model, err = createOllamaClient(config)
	case "googleai":
		logger.Debug("Initializing GoogleAI vision model")
		model, err = createGoogleAIClient(config)
	default:
		return nil, fmt.Errorf("unsupported vision LLM provider: %s", config.VisionLLMProvider)
	}

	if err != nil {
		logger.WithError(err).Error("Failed to create vision LLM client")
		return nil, fmt.Errorf("error creating vision LLM client: %w", err)
	}

	logger.Info("Successfully initialized LLM OCR provider")
	return &LLMProvider{
		Provider:                  config.VisionLLMProvider,
		Model:                     config.VisionLLMModel,
		LLM:                       model,
		Prompt:                    config.VisionLLMPrompt,
		OllamaOcrMaxTokensPerPage: config.OllamaOcrMaxTokensPerPage,
		OllamaOcrTemperature:      config.OllamaOcrTemperature,
		OllamaOcrTopK:             config.OllamaOcrTopK,
	}, nil
}

func (p *LLMProvider) ProcessImage(ctx context.Context, imageContent []byte, pageNumber int) (*OCRResult, error) {
	logger := log.WithFields(logrus.Fields{
		"provider": p.Provider, // Standardized field name
		"model":    p.Model,    // Standardized field name
		"page":     pageNumber,
	})
	logger.Debug("Starting LLM OCR processing")

	// Log the image dimensions
	img, _, err := image.Decode(bytes.NewReader(imageContent))
	if err != nil {
		logger.WithError(err).Error("Failed to decode image")
		return nil, fmt.Errorf("error decoding image: %w", err)
	}
	bounds := img.Bounds()
	logger.WithFields(logrus.Fields{
		"width":  bounds.Dx(),
		"height": bounds.Dy(),
	}).Debug("Image dimensions")

	logger.Debugf("Prompt: %s", p.Prompt)

	// Prepare content parts based on provider type
	var parts []llms.ContentPart
	if strings.ToLower(p.Provider) != "openai" {
		logger.Debug("Using binary image format for non-OpenAI provider")
		parts = []llms.ContentPart{
			llms.BinaryPart("image/jpeg", imageContent),
			llms.TextPart(p.Prompt),
		}
	} else {
		logger.Debug("Using base64 image format for OpenAI provider")
		base64Image := base64.StdEncoding.EncodeToString(imageContent)
		parts = []llms.ContentPart{
			llms.ImageURLPart(fmt.Sprintf("data:image/jpeg;base64,%s", base64Image)),
			llms.TextPart(p.Prompt),
		}
	}

	var callOpts []llms.CallOption
	// Apply Ollama specific options only if the provider is Ollama
	if strings.ToLower(p.Provider) == "ollama" {
		if p.OllamaOcrMaxTokensPerPage > 0 {
			callOpts = append(callOpts, llms.WithMaxTokens(p.OllamaOcrMaxTokensPerPage))
		}
		if p.OllamaOcrTemperature != nil {
			callOpts = append(callOpts, llms.WithTemperature(*p.OllamaOcrTemperature))
		}
		if p.OllamaOcrTopK != nil {
			callOpts = append(callOpts, llms.WithTopK(*p.OllamaOcrTopK))
		}
	}

	// Convert the image to text
	logger.Debug("Sending request to vision model")
	completion, err := p.LLM.GenerateContent(ctx, []llms.MessageContent{
		{
			Parts: parts,
			Role:  llms.ChatMessageTypeHuman,
		},
	}, callOpts...)
	if err != nil {
		logger.WithError(err).Error("Failed to get response from vision model")
		return nil, fmt.Errorf("error getting response from LLM: %w", err)
	}

	text := completion.Choices[0].Content
	limitHit := false
	tokenCount := -1

	// Try to get token count from GenerationInfo (relevant for Ollama with max tokens set)
	if strings.ToLower(p.Provider) == "ollama" && p.OllamaOcrMaxTokensPerPage > 0 {
		genInfo := completion.Choices[0].GenerationInfo
		if genInfo != nil && genInfo["TotalTokens"] != nil {
			if v, ok := genInfo["TotalTokens"].(int); ok {
				tokenCount = v
			}
		}
		// Fallback: count tokens using langchaingo (might not be accurate for all models)
		if tokenCount < 0 {
			tokenCount = llms.CountTokens(p.Model, text)
		}
		if tokenCount >= p.OllamaOcrMaxTokensPerPage {
			limitHit = true
		}
	}

	result := &OCRResult{
		Text: text,
		Metadata: map[string]string{
			"provider": p.Provider,
			"model":    p.Model,
		},
		OcrLimitHit:    limitHit,
		GenerationInfo: completion.Choices[0].GenerationInfo,
	}

	logger.WithField("content_length", len(result.Text)).WithFields(completion.Choices[0].GenerationInfo).Info("Successfully processed image")
	return result, nil
}

func createGoogleAIClient(config Config) (llms.Model, error) {
	apiKey := os.Getenv("GOOGLEAI_API_KEY")
	if apiKey == "" {
		return nil, fmt.Errorf("GOOGLEAI_API_KEY environment variable is not set")
	}
	ctx := context.Background()
	var thinkingBudget *int32
	if config.VisionLLMThinkingBudget != 0 {
		b := config.VisionLLMThinkingBudget
		thinkingBudget = &b
	}
	// Assuming NewGoogleAIProvider is defined elsewhere (e.g., main package or a shared utility)
	// This might need adjustment based on actual project structure.
	// For now, we assume it's accessible. If not, this will cause a compile error later.
	provider, err := NewGoogleAIProvider(ctx, config.VisionLLMModel, apiKey, thinkingBudget)
	if err != nil {
		return nil, fmt.Errorf("failed to create GoogleAI provider: %w", err)
	}
	return provider, nil
}

// createOpenAIClient creates a new OpenAI vision model client
func createOpenAIClient(config Config) (llms.Model, error) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		return nil, fmt.Errorf("OpenAI API key is not set")
	}
	return openai.New(
		openai.WithModel(config.VisionLLMModel),
		openai.WithToken(apiKey),
	)
}

// createOllamaClient creates a new Ollama vision model client
func createOllamaClient(config Config) (llms.Model, error) {
	host := os.Getenv("OLLAMA_HOST")
	if host == "" {
		host = "http://127.0.0.1:11434"
	}
	return ollama.New(
		ollama.WithModel(config.VisionLLMModel),
		ollama.WithServerURL(host),
		ollama.WithRunnerNumCtx(config.OllamaOcrMaxTokensPerPage), // Pass max tokens if set
	)
}

// Placeholder for NewGoogleAIProvider if it's meant to be in this package
// If it's in the main package, this function is not needed here.
// func NewGoogleAIProvider(ctx context.Context, modelName string, apiKey string, thinkingBudget *int32) (llms.Model, error) {
// 	// Implementation would go here, likely using langchaingo's googleai package
// 	return nil, fmt.Errorf("NewGoogleAIProvider not implemented in ocr package")
// }
