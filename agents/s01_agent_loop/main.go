package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/packages/param"
	"github.com/joho/godotenv"
)

var (
	client anthropic.Client
	model string
	system string
	tools = []anthropic.ToolUnionParam{
		anthropic.ToolUnionParam{OfTool: &anthropic.ToolParam{
			Name:        "bash",
			Description: param.Opt[string](anthropic.String("Run a bash command")),
			InputSchema: anthropic.ToolInputSchemaParam{
				Properties: map[string]map[string]string{
					"command": {"type": "string"},
				},
				Required: []string{"command"},
			},
			Type: anthropic.ToolTypeCustom,
		}}}
)

func init() {
	if err := godotenv.Overload(); err != nil {
		log.Fatal("Error loading .env file")
	}

	client = anthropic.NewClient()
	model = os.Getenv("MODEL_ID")
	dir, err := os.Getwd()
	if err != nil {
		log.Fatal("Error getting working directory")
	}
	system = fmt.Sprintf("You are a coding agent at %s. Use bash to solve task. Act, do not explain", dir)
}

func main() {
	var history []anthropic.MessageParam
	reader := bufio.NewReader(os.Stdin)
	fmt.Println("Enter a task: ")
	for {
		fmt.Print("-> ")
		text, err := reader.ReadString('\n')
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			log.Fatalf("input error: %v", err)
		}
		text = strings.Replace(text, "\n", "", -1)
		textLower := strings.ToLower(text)
		if textLower == "exit" || textLower == "q" || textLower == "" {
			break
		}
		history = append(history, anthropic.MessageParam{
			Role: "user",
			Content: []anthropic.ContentBlockParamUnion{
				anthropic.NewTextBlock(text),
			},
		})
		history = agentLoop(history)
		if len(history) >= 1 {
			lastResponse := history[len(history)-1]
			for _, content := range lastResponse.Content {
				if content.OfText != nil {
					fmt.Println(content.OfText.Text)
				}
			}
		}
	}
}

func agentLoop(messages []anthropic.MessageParam) []anthropic.MessageParam {
	for {
		response, err := client.Messages.New(context.Background(), anthropic.MessageNewParams{
			Model: anthropic.Model(model),
			System: []anthropic.TextBlockParam{
				{Text: system},
			},
			Messages:  messages,
			MaxTokens: 8000,
			Tools: tools,
		})
		if err != nil {
			log.Fatal("Error getting response: ", err)
		}
		for _, content := range response.Content {
			if content.Text != "" {
				messages = append(messages, anthropic.NewAssistantMessage(anthropic.NewTextBlock(content.Text)))
			}
		}
		if response.StopReason != anthropic.StopReasonToolUse {
			break
		}
		for _, content := range response.Content {
			if content.Type == "tool_use" {
				res := make(map[string]string)
				if err := json.Unmarshal(content.Input, &res); err != nil {
					log.Fatalf("Error unmarshalling input: %v, error: %v", content.Input, err)
				}
				cmd := res["command"]
				fmt.Println(cmd)
				output, isError, err := runBash(cmd)
				if err != nil {
					log.Fatalf("run bash %s failed, error: %v", cmd, err)
				}
				fmt.Println(output)
				messages = append(messages, anthropic.NewUserMessage(anthropic.NewToolResultBlock(content.ID, output, isError)))
			}
		}
	}
	return messages
}

func runBash(command string) (string, bool, error) {
	isError := false 
	dangerous := []string{"rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"}
	for _, d := range dangerous {
		if strings.Contains(d, command) {
			isError = true
			return "", isError, fmt.Errorf("Error: Dangerous command %s blocked", command)
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, "bash", "-c", command)
	current, _ := os.Getwd()
	cmd.Dir = current
	out, err := cmd.CombinedOutput()
	if err != nil {
		isError = true
		if ctx.Err() == context.DeadlineExceeded {
			return "", isError, fmt.Errorf("Error: Timeout (120s)")
		}
	}
	maxLength := 50000
	if len(out) >= maxLength {
		out = out[:maxLength]
	}
	output := string(out)
	if output == "" {
		output = "(no output)"
	}
	return output, isError, nil
}
