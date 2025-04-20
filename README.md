# OpenAI OCR MCP Server

A Model Context Protocol (MCP) server that provides OCR (Optical Character Recognition) functionality using OpenAI's vision capabilities. This server integrates with Cursor IDE to provide seamless text extraction from images.

## Features

- **Image Text Extraction**: Extract text from various image formats using OpenAI's GPT-4.1-mini vision model
- **Automatic Text File Creation**: Automatically saves extracted text alongside the source image
- **Content-Based File Naming**: Uses unique content hashing for organized file management
- **Multiple Image Format Support**: Supports JPG, PNG, GIF, and WebP formats
- **Robust Error Handling**: Comprehensive validation and error reporting
- **Detailed Logging**: Debug-friendly logging for troubleshooting

## Technical Details

### Vision Model
- Uses OpenAI's GPT-4.1-mini model
- Optimized for text extraction from images
- Supports high-detail image analysis
- Processes images through OpenAI's vision API

### File Processing
- Automatic text file creation
- Content-based hash generation
- Support for multiple image formats
- Built-in file size validation

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   npm install
   ```
3. Build the TypeScript code:
   ```bash
   npm run build
   ```
4. Set up your OpenAI API key in a `.env` file:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

### In Cursor IDE

1. Configure the MCP server in your Cursor settings
2. Use the OCR tool through Cursor's command palette
3. Select an image file to process
4. The extracted text will be:
   - Displayed in Cursor
   - Saved as a text file next to the image

### Text File Output

For each processed image, the server creates a text file with the following naming convention:

```
{original_image_name}-{content_hash}.txt
```

Example:
- Input image: `document.jpg`
- Output file: `document-a1b2c3d4.txt`

The `content_hash` is a unique 8-character hash generated from the extracted text, ensuring:
- Unique filenames for different text content
- Easy matching between source images and extracted text
- Version tracking when the same image produces different results

### Supported Image Formats

- JPEG/JPG
- PNG
- GIF
- WebP

### File Size Limits

- Maximum file size: 5MB
- Files exceeding this limit will be rejected with an error message

## Error Handling

The server provides detailed error messages for common issues:
- Invalid image formats
- File size exceeded
- File access problems
- API key issues
- Text extraction failures

## Development

### Building from Source

```bash
npm run build
```

### Running Tests

```bash
npm test
```

### Debugging

The server provides detailed logs including:
- API key validation
- File processing steps
- Text extraction results
- File saving operations

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- Supports both standard (`sk-...`) and project-specific (`sk-proj-...`) API keys

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](LICENSE)
