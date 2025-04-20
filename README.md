# OpenAI OCR MCP Server

A Model Context Protocol (MCP) server that provides OCR capabilities using OpenAI's vision API. This server allows you to extract text from images using OpenAI's powerful vision model.

## Prerequisites

- Node.js (v16 or higher)
- An OpenAI API key

## Environment Setup

This server requires an OpenAI API key to function. You must set it as an environment variable before running the server:

```bash
export OPENAI_API_KEY=your-api-key-here
```

You can obtain an API key from [OpenAI's platform](https://platform.openai.com/api-keys).

## Features

- Extract text from images using OpenAI's GPT-4o Vision model
- MCP-compatible service for integration with Cursor and other MCP clients
- Minimal implementation with no SDK dependencies
- Written in TypeScript with full type safety
- Communication via stdin/stdout for maximum compatibility

## How It Works

This server implements the Model Context Protocol (MCP) specification as a stdio-based service. Instead of using HTTP:

- The server reads JSON-RPC requests from stdin
- Processes them to extract text from images using OpenAI's vision API
- Writes JSON-RPC responses to stdout

This approach makes it compatible with Cursor's MCP client system and other tools that can communicate via stdio pipes.

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   npm install
   ```
3. Build the project:
   ```bash
   npm run build
   ```

## Usage

The server is designed to work with Cursor IDE through the Model Context Protocol (MCP). Once properly configured in Cursor, you can use it to extract text from images.

### Tool Capabilities

The server provides the following tool:

- `extract_text_from_image`: Extracts text from a local image file
  - Parameter: `image_path` (string) - Full path to a local image file
  - Supported formats: .jpg, .jpeg, .png, .gif, .webp
  - Maximum file size: 5MB

### Error Messages

If you see an error about the OpenAI API key not being set, make sure you've set the environment variable correctly:

```bash
export OPENAI_API_KEY=your-api-key-here
```

Note: You'll need to restart Cursor after setting the environment variable for it to take effect.

## Development

To build the TypeScript code:
```bash
npm run build
```

To run tests:
```bash
npm test
```

## Available Tools

- **extract_text_from_image**: Extract text from an image file
  - Parameters:
    - `image_path`: Path to a local image file
  - Returns: Extracted text content

## Supported Image Types

- JPEG (.jpg, .jpeg)
- PNG (.png)
- GIF (.gif)
- WebP (.webp)

## File Size Limits

The maximum file size is 5MB.

## Implementation Details

This server was originally implemented using the MCP SDK with an HTTP/SSE approach. However, we've moved to a simpler stdio-based approach for better compatibility with Cursor.

The current implementation:

- Is pure TypeScript with comprehensive type definitions
- Makes OpenAI API calls using curl to avoid additional dependencies
- Logs detailed information to stderr for debugging
- Precisely matches the JSON-RPC protocol format used by MCP

Logs and debug information are written to stderr, while JSON-RPC responses are written to stdout, ensuring clean communication via stdio pipes.

## License

[MIT](LICENSE)
