#!/usr/bin/env node

/**
 * OpenAI OCR MCP Server
 * A stdio-based Model Context Protocol (MCP) server for OCR functionality
 * using OpenAI's vision capabilities.
 * 
 * This server:
 * 1. Accepts JSON-RPC requests via stdin
 * 2. Processes images using OpenAI's GPT-4.1-mini vision model
 * 3. Returns extracted text and analysis via stdout
 * 4. Automatically saves extracted text to a file alongside the image
 *    in the format: {image_name}-{hash}.txt where {hash} is a unique
 *    8-character hash of the extracted text content
 * 5. Automatically generates and appends AI analysis of the extracted text
 * 6. Provides tools for both text extraction and analysis management
 * 
 * Available Tools:
 * - extract_text_from_image: Extract and analyze text from images
 * - append_analysis: Append additional AI analysis to existing OCR text files
 * 
 * Model Information:
 * - Uses GPT-4.1-mini for vision tasks
 * - Optimized for text extraction and analysis
 * - Supports high-detail image analysis
 * 
 * Features:
 * - Image validation (size, format, accessibility)
 * - Robust error handling and logging
 * - Automatic text file creation with content hashing
 * - Automatic AI analysis generation and storage
 * - Support for both standard and project-specific OpenAI API keys
 * - LLM response tracking and management
 * 
 * File Naming Example:
 * - Input image: /path/to/scan.jpg
 * - Output text: /path/to/scan-a1b2c3d4.txt
 *   where 'a1b2c3d4' is a unique hash of the extracted text
 * 
 * File Format:
 * The output text file contains:
 * 1. OCR EXTRACTED TEXT section with raw text from the image
 * 2. LLM ANALYSIS section(s) with AI-generated analysis
 *    - Initial analysis generated during extraction
 *    - Additional analyses can be appended using the append_analysis tool
 * 
 * Environment Setup:
 * - Requires OpenAI API key in environment variables
 * - Supports multiple API key formats (OPENAI_API_KEY, openai_api_key)
 * - Optional .env file support for API key configuration
 */

// Core imports
import * as fs from 'fs';
import * as path from 'path';
import { createInterface } from 'readline';
import { spawn, ChildProcessWithoutNullStreams } from 'child_process';
import * as dotenv from 'dotenv';
import { createHash } from 'crypto';

// ============================================================================
// Configuration
// ============================================================================

// Global API key variable
let OPENAI_API_KEY: string | undefined;

// Add conversation tracking
interface ConversationState {
  lastOcrResult?: string;
  lastImagePath?: string;
  llmResponses: Array<{
    timestamp: string;
    response: string;
  }>;
}

// Initialize conversation state
const conversationState: ConversationState = {
  llmResponses: []
};

/**
 * Validate the API key format and check for common issues
 */
function validateApiKey(key: string): { isValid: boolean; issue?: string } {
  if (!key) {
    return { isValid: false, issue: 'API key is empty' };
  }

  // Remove any whitespace or newlines that might have been added
  const cleanKey = key.trim();
  
  // Check for truncation issues
  if (cleanKey === 'sk-proj-' || cleanKey.startsWith('sk-proj-*')) {
    return { isValid: false, issue: 'API key appears to be truncated' };
  }

  // Check for proper format and length
  const isStandardKey = cleanKey.startsWith('sk-') && cleanKey.length > 20;
  const isProjectKey = cleanKey.startsWith('sk-proj-') && cleanKey.length > 30;
  
  if (!isStandardKey && !isProjectKey) {
    return { 
      isValid: false, 
      issue: `Invalid key format. Key must start with "sk-" or "sk-proj-" and be of sufficient length. Got: ${cleanKey.substring(0, 8)}...`
    };
  }

  return { isValid: true };
}

/**
 * Get OpenAI API key from environment variables, checking multiple formats
 */
function getOpenAIApiKey(): string | undefined {
  // Check for various possible environment variable names and formats
  const possibleEnvVars = [
    'OPENAI_API_KEY',
    'openai_api_key',
    'OpenAI_API_Key'
  ];
  
  for (const envVar of possibleEnvVars) {
    const key = process.env[envVar];
    if (key && key !== 'your-api-key-here') {
      // Clean the key and validate it
      const cleanKey = key.trim();
      const validation = validateApiKey(cleanKey);
      
      if (validation.isValid) {
        log(`Found valid API key in environment variable: ${envVar}`);
        return cleanKey;
      } else {
        log(`Found API key in ${envVar} but it was invalid: ${validation.issue}`);
      }
    }
  }
  
  return undefined;
}

/**
 * Debug function to check environment variables
 */
async function debugEnvironment(): Promise<void> {
  log('Checking environment configuration...');
  
  // Check all possible environment variable names
  const possibleEnvVars = [
    'OPENAI_API_KEY',
    'openai_api_key',
    'OpenAI_API_Key'
  ];
  
  for (const envVar of possibleEnvVars) {
    log(`Checking for ${envVar}: ${!!process.env[envVar]}`);
  }
  
  // Check environment using the env command
  const envProcess = spawn('env');
  let envOutput = '';
  
  envProcess.stdout.on('data', (data) => {
    envOutput += data.toString();
  });
  
  await new Promise((resolve) => {
    envProcess.on('close', () => {
      // Log environment but mask any API keys
      const maskedEnv = envOutput.split('\n').map(line => {
        if (line.toLowerCase().includes('api_key')) {
          return line.replace(/=.+$/, '=*****');
        }
        return line;
      }).join('\n');
      resolve(undefined);
    });
  });
}

// Constants
const MAX_FILE_SIZE = 5 * 1024 * 1024; // 5MB
const ALLOWED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.webp'];
const SERVER_NAME = "openai-ocr-service";
const SERVER_VERSION = "1.0.0";
const PROTOCOL_VERSION = "2024-11-05"; // Updated to match MCP Inspector expectation

// Store client info for future use
let clientInfo: any = null;

// ============================================================================
// Types
// ============================================================================

// JSON-RPC Protocol Types
interface JsonRpcRequest {
  jsonrpc: string;
  id: string | number;
  method: string;
  params?: any;
}

interface JsonRpcSuccessResponse {
  jsonrpc: string;
  id: string | number;
  result: any;
}

interface JsonRpcErrorResponse {
  jsonrpc: string;
  id: string | number | null;
  error: {
    code: number;
    message: string;
  };
}

type JsonRpcResponse = JsonRpcSuccessResponse | JsonRpcErrorResponse;

// MCP Protocol Types
interface McpToolParameter {
  type: string;
  properties: {
    [key: string]: {
      type: string;
      description: string;
    };
  };
  required?: string[];
}

interface McpTool {
  name: string;
  description: string;
  parameters: McpToolParameter;
}

interface McpServerInfo {
  name: string;
  version: string;
}

interface McpOfferings {
  tools: McpTool[];
}

interface McpCapabilities {
  tools: {
    [key: string]: {
      description: string;
      parameters: McpToolParameter;
    };
  };
}

interface McpInitializeResponse {
  protocolVersion: string;
  serverInfo: McpServerInfo;
  clientInfo?: any;
  capabilities: McpCapabilities;
}

interface McpListOfferingsResponse {
  protocolVersion: string;
  serverInfo: McpServerInfo;
  clientInfo?: any;
  offerings: McpOfferings;
}

interface McpContentItem {
  type: string;
  text: string;
}

interface McpToolResult {
  content: McpContentItem[];
  isError?: boolean;
}

// OCR-specific Types
interface OcrParams {
  image_path: string;
}

interface OpenAIMessage {
  role: string;
  content: any[];
}

interface OpenAIRequest {
  model: string;
  messages: OpenAIMessage[];
  max_tokens: number;
}

interface OpenAIResponse {
  choices: Array<{
    message: {
      content: string;
    };
  }>;
  error?: {
    message: string;
  };
}

// ============================================================================
// Tool Definition
// ============================================================================

/**
 * The OCR tool specification
 */
const OCR_TOOL: McpTool = {
  name: "extract_text_from_image",
  description: "Extract text from images using OpenAI's vision capabilities. Simply provide the full path to a local image file.",
  parameters: {
    type: "object",
    properties: {
      image_path: {
        type: "string",
        description: "Full path to a local image file (e.g., /Users/username/Pictures/image.jpg)"
      }
    },
    required: ["image_path"]
  }
};

// Add new tool definition
const APPEND_ANALYSIS_TOOL: McpTool = {
  name: "append_analysis",
  description: "Append LLM analysis to an OCR text file. This tool is used to add AI commentary to existing OCR results.",
  parameters: {
    type: "object",
    properties: {
      text_file_path: {
        type: "string",
        description: "Path to the OCR text file to append analysis to"
      },
      analysis: {
        type: "string",
        description: "The LLM's analysis to append to the file"
      }
    },
    required: ["text_file_path", "analysis"]
  }
};

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Set up logging to stderr
 */
function log(message: string): void {
  console.error(`[${new Date().toISOString()}] ${message}`);
}

// Log initial configuration status
log('OpenAI OCR MCP Server starting up...');

// Initialize environment variables from .env file
try {
  const envResult = dotenv.config();
  if (envResult.error) {
    log('Warning: Error loading .env file: ' + envResult.error.message);
    log('Will attempt to use environment variables from system');
  } else {
    log('Successfully loaded .env file');
  }
} catch (error) {
  log('Warning: Error loading .env file: ' + (error instanceof Error ? error.message : String(error)));
  log('Will attempt to use environment variables from system');
}

// Initialize API key
OPENAI_API_KEY = getOpenAIApiKey();
if (!OPENAI_API_KEY) {
  log('ERROR: Valid OpenAI API key not found in environment variables or .env file');
  log('Please set the API key in your .env file:');
  log('OPENAI_API_KEY=your_api_key_here');
  log('or');
  log('openai_api_key=your_api_key_here');
  log('');
  log('Make sure:');
  log('1. The .env file exists in the project root directory');
  log('2. There are no spaces around the equals sign');
  log('3. The API key is not wrapped in quotes');
  log('4. The API key starts with either "sk-" or "sk-proj-"');
}

// Debug environment configuration
debugEnvironment().then(() => {
  const keyStatus = OPENAI_API_KEY 
    ? `Found and loaded (starts with: ${OPENAI_API_KEY.substring(0, 8)}...)`
    : 'Not found';
  log(`API Key status: ${keyStatus}`);
});

/**
 * Send a JSON-RPC response to stdout
 */
function sendResponse(id: string | number, result: any): void {
  const response: JsonRpcSuccessResponse = {
    jsonrpc: '2.0',
    id,
    result
  };
  process.stdout.write(JSON.stringify(response) + '\n');
  log(`Sent response for id: ${id}`);
}

/**
 * Send a JSON-RPC error to stdout
 */
function sendError(id: string | number | null, code: number, message: string): void {
  const response: JsonRpcErrorResponse = {
    jsonrpc: '2.0',
    id: id || 'error',
    error: {
      code,
      message
    }
  };
  process.stdout.write(JSON.stringify(response) + '\n');
  log(`Sent error for id: ${id}, code: ${code}, message: ${message}`);
}

/**
 * Helper function to determine MIME type
 */
function getMimeType(filePath: string): string {
  const ext = path.extname(filePath).toLowerCase();
  switch (ext) {
    case '.jpg':
    case '.jpeg':
      return 'image/jpeg';
    case '.png':
      return 'image/png';
    case '.gif':
      return 'image/gif';
    case '.webp':
      return 'image/webp';
    default:
      return 'image/jpeg';  // Default to JPEG
  }
}

/**
 * Generate a short hash of text content
 */
function generateShortHash(text: string): string {
  // Create a hash of the text content
  const hash = createHash('sha256').update(text).digest('hex');
  // Return first 8 characters of the hash
  return hash.substring(0, 8);
}

/**
 * Save extracted text to a file with hash in the filename
 */
function saveExtractedText(imagePath: string, text: string): string {
  // Get the directory and filename without extension
  const dir = path.dirname(imagePath);
  const baseNameWithoutExt = path.basename(imagePath, path.extname(imagePath));
  
  // Generate hash of the text content
  const hash = generateShortHash(text);
  
  // Create filename with format: {image_name}-{hash}.txt
  const txtPath = path.join(dir, `${baseNameWithoutExt}-${hash}.txt`);

  try {
    // Save the initial OCR text
    fs.writeFileSync(txtPath, `OCR EXTRACTED TEXT:\n==================\n${text}\n`, 'utf8');
    log(`Successfully saved extracted text to: ${txtPath}`);
    return txtPath;
  } catch (error) {
    log(`Error saving text file: ${error instanceof Error ? error.message : String(error)}`);
    throw new Error(`Failed to save text file: ${error instanceof Error ? error.message : String(error)}`);
  }
}

/**
 * Append LLM response to the text file
 */
function appendLlmResponseToFile(txtPath: string, response: any): void {
  try {
    // Format the LLM response section
    const timestamp = new Date().toISOString();
    const separator = '\n\nLLM ANALYSIS:\n=============\n';
    const formattedResponse = typeof response === 'string' ? response : JSON.stringify(response, null, 2);
    const appendText = `${separator}[${timestamp}]\n${formattedResponse}\n`;

    // Append to the file
    fs.appendFileSync(txtPath, appendText, 'utf8');
    log(`Successfully appended LLM response to: ${txtPath}`);
  } catch (error) {
    log(`Error appending LLM response: ${error instanceof Error ? error.message : String(error)}`);
  }
}

/**
 * Log LLM response to the OCR result and append to text file
 */
function logLlmResponse(response: any): void {
  const timestamp = new Date().toISOString();
  log(`[LLM Response at ${timestamp}]`);
  log(`Previous OCR image: ${conversationState.lastImagePath}`);
  log(`LLM commented on OCR result: ${JSON.stringify(response, null, 2)}`);
  
  // Store in conversation state
  conversationState.llmResponses.push({
    timestamp,
    response: JSON.stringify(response)
  });

  // If we have the last image path, try to find and update the corresponding text file
  if (conversationState.lastImagePath) {
    try {
      const dir = path.dirname(conversationState.lastImagePath);
      const baseNameWithoutExt = path.basename(conversationState.lastImagePath, path.extname(conversationState.lastImagePath));
      
      log(`Looking for text file in directory: ${dir}`);
      log(`Searching for files starting with: ${baseNameWithoutExt}-`);
      
      // Read the directory to find the matching text file
      const files = fs.readdirSync(dir);
      log(`Found ${files.length} files in directory`);
      
      const matchingFiles = files.filter(file => 
        file.startsWith(baseNameWithoutExt + '-') && 
        file.endsWith('.txt')
      );
      log(`Found ${matchingFiles.length} matching text files: ${JSON.stringify(matchingFiles)}`);

      if (matchingFiles.length > 0) {
        // Use the most recently created file if there are multiple matches
        const matchingFile = matchingFiles[matchingFiles.length - 1];
        const txtPath = path.join(dir, matchingFile);
        log(`Selected text file to update: ${txtPath}`);
        
        // Read current content for debugging
        const currentContent = fs.readFileSync(txtPath, 'utf8');
        log(`Current file content (first 100 chars): ${currentContent.substring(0, 100)}...`);
        
        appendLlmResponseToFile(txtPath, response);
        
        // Verify the append operation
        const updatedContent = fs.readFileSync(txtPath, 'utf8');
        log(`Updated file content (last 100 chars): ${updatedContent.substring(updatedContent.length - 100)}...`);
      } else {
        log(`Could not find corresponding text file for image: ${conversationState.lastImagePath}`);
        log(`Available files in directory: ${JSON.stringify(files)}`);
      }
    } catch (error) {
      log(`Error handling LLM response file update: ${error instanceof Error ? error.message : String(error)}`);
      if (error instanceof Error && error.stack) {
        log(`Stack trace: ${error.stack}`);
      }
    }
  } else {
    log('No previous image path found in conversation state');
  }
}

// ============================================================================
// Core Tool Implementation
// ============================================================================

/**
 * Implementation of the extract_text_from_image tool
 */
async function extractTextFromImage(args: OcrParams): Promise<McpToolResult> {
  try {
    // First verify we have an API key
    if (!OPENAI_API_KEY) {
      // Try to get it again in case it was set after startup
      OPENAI_API_KEY = getOpenAIApiKey();
      if (!OPENAI_API_KEY) {
        throw new Error('OpenAI API key is not available. Please set the environment variable.');
      }
    }

    // Validate API key format
    const validation = validateApiKey(OPENAI_API_KEY);
    if (!validation.isValid) {
      throw new Error(`Invalid API key: ${validation.issue}`);
    }

    log(`Using API key: ${OPENAI_API_KEY}`);
    log(`Using API key format: ${OPENAI_API_KEY.startsWith('sk-proj-') ? 'Project-specific' : 'Standard'}`);
    log(`API key length: ${OPENAI_API_KEY.length} characters`);
    log(`API key prefix: ${OPENAI_API_KEY.substring(0, 8)}...`);

    // Validate args
    if (!args) {
      throw new Error("Arguments are required but were undefined");
    }
    
    const { image_path } = args;
    
    // Validate image_path
    if (!image_path) {
      throw new Error("image_path is required but was missing");
    }
    
    log(`Processing OCR request for path: ${image_path}`);
    
    // Resolve to absolute path and log it
    const imagePath = path.resolve(image_path);
    log(`Resolved absolute path: ${imagePath}`);
    
    // Detailed file validation
    log('Validating file...');
    
    try {
      const exists = fs.existsSync(imagePath);
      log(`File exists check: ${exists}`);
      if (!exists) {
        throw new Error(`File not found at path: ${imagePath}`);
      }
    } catch (error) {
      log(`Error checking file existence: ${error instanceof Error ? error.message : String(error)}`);
      throw new Error(`File access error: ${error instanceof Error ? error.message : String(error)}`);
    }
    
    try {
      const stats = fs.statSync(imagePath);
      log(`File stats: isFile=${stats.isFile()}, size=${stats.size} bytes`);
      
      if (!stats.isFile()) {
        throw new Error(`Not a file: ${imagePath}`);
      }
      
      if (stats.size > MAX_FILE_SIZE) {
        throw new Error(`File too large: ${(stats.size / (1024 * 1024)).toFixed(2)}MB (max: ${MAX_FILE_SIZE / (1024 * 1024)}MB)`);
      }
    } catch (error) {
      log(`Error checking file stats: ${error instanceof Error ? error.message : String(error)}`);
      throw new Error(`File stats error: ${error instanceof Error ? error.message : String(error)}`);
    }
    
    const ext = path.extname(imagePath).toLowerCase();
    log(`File extension: ${ext}`);
    if (!ALLOWED_EXTENSIONS.includes(ext)) {
      throw new Error(`Invalid file type: ${ext}. Allowed types: ${ALLOWED_EXTENSIONS.join(', ')}`);
    }
    
    // Try to read a small portion of the file to verify access
    try {
      const fd = fs.openSync(imagePath, 'r');
      const buffer = Buffer.alloc(1024);
      fs.readSync(fd, buffer, 0, 1024, 0);
      fs.closeSync(fd);
      log('Successfully verified file read access');
    } catch (error) {
      log(`Error reading file: ${error instanceof Error ? error.message : String(error)}`);
      throw new Error(`File read error: ${error instanceof Error ? error.message : String(error)}`);
    }
    
    // Read the image file
    log(`Reading full image file: ${imagePath}`);
    let imageBuffer: Buffer;
    try {
      imageBuffer = fs.readFileSync(imagePath);
      log(`Successfully read ${imageBuffer.length} bytes`);
    } catch (error) {
      log(`Error reading full file: ${error instanceof Error ? error.message : String(error)}`);
      throw new Error(`Failed to read image file: ${error instanceof Error ? error.message : String(error)}`);
    }
    
    const base64Image = Buffer.from(imageBuffer).toString('base64');
    const mimeType = getMimeType(imagePath);
    const imageUrl = `data:${mimeType};base64,${base64Image}`;
    
    // Call OpenAI API using curl
    log(`Calling OpenAI API for image of size ${imageBuffer.length} bytes`);
    
    return new Promise<McpToolResult>((resolve, reject) => {
      const requestBody: OpenAIRequest = {
        model: "gpt-4.1-mini",
        messages: [
          {
            role: 'user',
            content: [
              { 
                type: 'text', 
                text: 'Extract all text from this image and provide a detailed analysis of its contents. Include both the raw text and your analysis.' 
              },
              { 
                type: 'image_url',
                image_url: {
                  url: imageUrl,
                  detail: 'high'
                }
              }
            ]
          }
        ],
        max_tokens: 4096
      };
      
      log('Sending request to OpenAI API...');
      log(`Using API key found in environment (length: ${OPENAI_API_KEY?.length})`);
      
      const curl: ChildProcessWithoutNullStreams = spawn('curl', [
        '-s',
        '-X', 'POST',
        'https://api.openai.com/v1/chat/completions',
        '-H', `Authorization: Bearer ${OPENAI_API_KEY}`,
        '-H', 'Content-Type: application/json',
        '-v', // Add verbose output
        '-d', JSON.stringify(requestBody)
      ]);
      
      let responseData = '';
      let errorOutput = '';
      
      curl.stdout.on('data', (data: Buffer) => {
        responseData += data.toString();
        log(`API Response data: ${data.toString()}`);
      });
      
      curl.stderr.on('data', (data: Buffer) => {
        errorOutput += data.toString();
        // Log the full error output for debugging
        log(`curl stderr (full): ${data.toString()}`);
      });
      
      curl.on('close', async (code: number) => {
        if (code !== 0) {
          log(`curl process exited with code ${code}`);
          log(`Full error output: ${errorOutput}`);
          log(`API Key being used (first 10 chars): ${OPENAI_API_KEY?.substring(0, 10)}...`);
          reject(new Error(`curl process exited with code ${code}: ${errorOutput}`));
          return;
        }
        
        try {
          log(`Full OpenAI API response: ${responseData}`);
          const response = JSON.parse(responseData) as OpenAIResponse;
          if (response.error) {
            log(`OpenAI API error details: ${JSON.stringify(response.error, null, 2)}`);
            reject(new Error(`OpenAI API error: ${response.error.message}`));
            return;
          }
          
          const fullResponse = response.choices[0].message.content || "No text extracted";
          
          // Split the response into extracted text and analysis
          const parts = fullResponse.split(/\n\n(?=Analysis:|Interpretation:|Summary:|Understanding:)/i);
          const extractedText = parts[0];
          const analysis = parts.length > 1 ? parts.slice(1).join('\n\n') : '';
          
          log(`Successfully extracted ${extractedText.length} characters of text`);
          
          // Store the OCR result in conversation state
          conversationState.lastOcrResult = extractedText;
          conversationState.lastImagePath = imagePath;
          
          // Save the extracted text to a file
          try {
            const txtPath = saveExtractedText(imagePath, extractedText);
            log(`Text file created: ${txtPath}`);
            
            // If we have analysis, append it using our tool
            if (analysis) {
              try {
                await handleAppendAnalysis({
                  text_file_path: txtPath,
                  analysis: analysis
                });
                log('Successfully appended initial analysis');
              } catch (error) {
                log(`Warning: Failed to append initial analysis: ${error}`);
              }
            }
            
            resolve({ 
              content: [
                { type: "text", text: extractedText },
                { type: "text", text: `\n\nText has been saved to: ${txtPath}` }
              ]
            });
          } catch (error) {
            // If saving fails, still return the extracted text but include the error
            log(`Warning: Failed to save text file: ${error instanceof Error ? error.message : String(error)}`);
            resolve({ 
              content: [
                { type: "text", text: extractedText },
                { type: "text", text: `\n\nWarning: Failed to save text file: ${error instanceof Error ? error.message : String(error)}` }
              ]
            });
          }
        } catch (error) {
          log(`Error parsing OpenAI response: ${error instanceof Error ? error.message : String(error)}`);
          reject(new Error(`Failed to parse OpenAI response: ${error instanceof Error ? error.message : String(error)}`));
        }
      });
    });
  } catch (error) {
    log(`Error in extractTextFromImage: ${error instanceof Error ? error.message : String(error)}`);
    return { 
      content: [
        { type: "text", text: `Error: ${error instanceof Error ? error.message : String(error)}` }
      ],
      isError: true
    };
  }
}

// ============================================================================
// MCP Method Handlers
// ============================================================================

/**
 * Handle the initialize method
 */
function handleInitialize(id: string | number, params?: any): void {
  log('Handling initialize request');
  
  // Log client protocol version if provided
  if (params && params.protocolVersion) {
    log(`Client protocol version: ${params.protocolVersion}`);
  }
  
  // Log client capabilities if provided
  if (params && params.capabilities) {
    log(`Client capabilities: ${JSON.stringify(params.capabilities, null, 2)}`);
  }
  
  // Store client info for future use
  if (params && params.clientInfo) {
    clientInfo = params.clientInfo;
    log(`Client info: ${JSON.stringify(clientInfo, null, 2)}`);
  }
  
  // Create the response
  const response: McpInitializeResponse = {
    protocolVersion: PROTOCOL_VERSION,
    serverInfo: {
      name: SERVER_NAME,
      version: SERVER_VERSION
    },
    capabilities: {
      tools: {
        [OCR_TOOL.name]: {
          description: OCR_TOOL.description,
          parameters: OCR_TOOL.parameters
        },
        [APPEND_ANALYSIS_TOOL.name]: {
          description: APPEND_ANALYSIS_TOOL.description,
          parameters: APPEND_ANALYSIS_TOOL.parameters
        }
      }
    }
  };
  
  // Include client info in response if provided
  if (clientInfo) {
    response.clientInfo = clientInfo;
  }
  
  sendResponse(id, response);
}

/**
 * Handle the ListOfferings method
 */
function handleListOfferings(id: string | number): void {
  log('Handling ListOfferings request');
  
  // Create the response
  const response: McpListOfferingsResponse = {
    protocolVersion: PROTOCOL_VERSION,
    serverInfo: {
      name: SERVER_NAME,
      version: SERVER_VERSION
    },
    offerings: {
      tools: [OCR_TOOL, APPEND_ANALYSIS_TOOL]
    }
  };
  
  // Include client info in response if available
  if (clientInfo) {
    response.clientInfo = clientInfo;
  }
  
  sendResponse(id, response);
}

/**
 * Handle the callTool method
 */
async function handleCallTool(id: string | number, params: any): Promise<void> {
  log(`Handling callTool request: ${JSON.stringify(params)}`);

  if (!params) {
    sendError(id, -32602, "Invalid params");
    return;
  }

  const { name, arguments: args } = params;

  if (name === OCR_TOOL.name) {
    try {
      const result = await extractTextFromImage(args as OcrParams);
      sendResponse(id, result);
    } catch (error) {
      log(`Error executing tool: ${error instanceof Error ? error.message : String(error)}`);
      sendError(id, -32000, `Error executing tool: ${error instanceof Error ? error.message : String(error)}`);
    }
  } else if (name === APPEND_ANALYSIS_TOOL.name) {
    try {
      const result = await handleAppendAnalysis(args as { text_file_path: string; analysis: string });
      sendResponse(id, result);
    } catch (error) {
      log(`Error executing tool: ${error instanceof Error ? error.message : String(error)}`);
      sendError(id, -32000, `Error executing tool: ${error instanceof Error ? error.message : String(error)}`);
    }
  } else {
    log(`Tool not found: ${name}`);
    sendError(id, -32601, `Tool not found: ${name}`);
  }
}

/**
 * Handle notification methods (methods that start with 'notifications/')
 * These don't require a response
 */
function handleNotification(method: string, params?: any): void {
  log(`Handling notification: ${method}`);
  log(`Notification params: ${JSON.stringify(params, null, 2)}`);
  
  switch (method) {
    case 'notifications/llm_response':
      log('Received LLM response notification');
      // First log the response as usual
      logLlmResponse(params);
      
      // Then use our append_analysis tool if we have a valid text file path
      if (conversationState.lastOcrResult && conversationState.lastImagePath) {
        const dir = path.dirname(conversationState.lastImagePath);
        const baseNameWithoutExt = path.basename(conversationState.lastImagePath, path.extname(conversationState.lastImagePath));
        
        // Find the corresponding text file
        try {
          const files = fs.readdirSync(dir);
          const matchingFiles = files.filter(file => 
            file.startsWith(baseNameWithoutExt + '-') && 
            file.endsWith('.txt')
          );
          
          if (matchingFiles.length > 0) {
            // Use the most recently created file
            const txtPath = path.join(dir, matchingFiles[matchingFiles.length - 1]);
            
            // Call our append_analysis tool
            handleAppendAnalysis({
              text_file_path: txtPath,
              analysis: typeof params === 'string' ? params : JSON.stringify(params, null, 2)
            }).catch(error => {
              log(`Error appending analysis: ${error}`);
            });
          }
        } catch (error) {
          log(`Error finding text file: ${error}`);
        }
      }
      break;
    case 'notifications/initialized':
      log('Client has been fully initialized');
      break;
    case 'notifications/exit':
      log('Client has requested exit');
      break;
    default:
      log(`Unknown notification method: ${method}`);
  }
}

/**
 * Handle the tools/list method which is used by the MCP Inspector
 */
function handleToolsList(id: string | number): void {
  log('Handling tools/list request');
  
  // Create the response with the required schemas
  const response = {
    tools: [
      {
        name: OCR_TOOL.name,
        description: OCR_TOOL.description,
        inputSchema: {
          type: "object",
          properties: {
            image_path: {
              type: "string",
              description: "Full path to a local image file (e.g., /Users/username/Pictures/image.jpg)"
            }
          },
          required: ["image_path"]
        },
        outputSchema: {
          type: "object",
          properties: {
            content: {
              type: "array",
              items: {
                type: "object",
                properties: {
                  type: {
                    type: "string",
                    description: "Type of content (text)"
                  },
                  text: {
                    type: "string",
                    description: "Extracted text from the image"
                  }
                }
              }
            }
          }
        }
      },
      {
        name: APPEND_ANALYSIS_TOOL.name,
        description: APPEND_ANALYSIS_TOOL.description,
        inputSchema: {
          type: "object",
          properties: {
            text_file_path: {
              type: "string",
              description: "Path to the OCR text file to append analysis to"
            },
            analysis: {
              type: "string",
              description: "The LLM's analysis to append to the file"
            }
          },
          required: ["text_file_path", "analysis"]
        },
        outputSchema: {
          type: "object",
          properties: {
            content: {
              type: "array",
              items: {
                type: "object",
                properties: {
                  type: {
                    type: "string",
                    description: "Type of content (text)"
                  },
                  text: {
                    type: "string",
                    description: "Status message about the analysis being appended"
                  }
                }
              }
            }
          }
        }
      }
    ]
  };
  
  sendResponse(id, response);
}

/**
 * Handle the tools/call method which is used by the MCP Inspector to call tools
 */
async function handleToolsCall(id: string | number, params: any): Promise<void> {
  log(`Handling tools/call request: ${JSON.stringify(params)}`);
  
  if (!params) {
    sendError(id, -32602, "Invalid params: params is undefined");
    return;
  }
  
  // MCP Inspector might send the parameters directly or in different formats
  // Try to extract the tool name and parameters in several ways
  
  let toolName: string | undefined;
  let imagePath: string | undefined;
  
  // Case 1: If params is a simple string, assume it's the image_path directly
  if (typeof params === 'string') {
    toolName = OCR_TOOL.name;
    imagePath = params;
    log(`Received direct string parameter: ${imagePath}`);
  }
  // Case 2: { name, params } format with string params
  else if (params.name && typeof params.params === 'string') {
    toolName = params.name;
    imagePath = params.params;
    log(`Received name with string params: ${toolName}, ${imagePath}`);
  }
  // Case 3: { name, params: { image_path } } format
  else if (params.name && params.params && params.params.image_path) {
    toolName = params.name;
    imagePath = params.params.image_path;
    log(`Received name with object params: ${toolName}, ${imagePath}`);
  }
  // Case 4: Direct object with image_path
  else if (params.image_path) {
    toolName = OCR_TOOL.name;
    imagePath = params.image_path;
    log(`Received direct object with image_path: ${imagePath}`);
  }
  // Case 5: { name, arguments } format
  else if (params.name && params.arguments) {
    toolName = params.name;
    // If arguments is a string, use it directly
    if (typeof params.arguments === 'string') {
      imagePath = params.arguments;
      log(`Received string arguments: ${imagePath}`);
    }
    // If arguments is an object, look for image_path
    else if (params.arguments.image_path) {
      imagePath = params.arguments.image_path;
      log(`Received object arguments with image_path: ${imagePath}`);
    }
  }
  
  // Log details for debugging
  log(`Parsed tool call - name: ${toolName}, imagePath: ${imagePath}`);
  
  if (!toolName) {
    sendError(id, -32602, `Invalid params: tool name could not be determined from params: ${JSON.stringify(params)}`);
    return;
  }
  
  if (!imagePath) {
    sendError(id, -32602, `Invalid params: Please provide a path to a local image file as a simple string parameter`);
    return;
  }
  
  if (toolName === OCR_TOOL.name) {
    try {
      // Create the params object for our tool function
      const toolParams: OcrParams = { image_path: imagePath };
      
      const result = await extractTextFromImage(toolParams);
      sendResponse(id, result);
    } catch (error) {
      log(`Error executing tool: ${error instanceof Error ? error.message : String(error)}`);
      sendError(id, -32000, `Error executing tool: ${error instanceof Error ? error.message : String(error)}`);
    }
  } else {
    log(`Tool not found: ${toolName}`);
    sendError(id, -32601, `Tool not found: ${toolName}`);
  }
}

// ============================================================================
// Server Management
// ============================================================================

/**
 * Add a keep-alive function that runs indefinitely
 */
function keepAlive(): void {
  const interval = setInterval(() => {
    log('Keep-alive heartbeat...');
  }, 30000); // Send a heartbeat log every 30 seconds
  
  // Prevent the interval from keeping the process alive if everything else ends
  interval.unref();
  
  // Ensure we clean up properly on exit
  process.on('exit', () => {
    log('Process is exiting, clearing keep-alive interval');
    clearInterval(interval);
  });
  
  log('Keep-alive mechanism activated');
}

/**
 * Set up process error handlers
 */
function setupErrorHandlers(): void {
  process.on('uncaughtException', (error: Error) => {
    log(`UNCAUGHT EXCEPTION: ${error.message}`);
    log(error.stack || '');
  });

  process.on('unhandledRejection', (reason: unknown) => {
    log(`UNHANDLED PROMISE REJECTION: ${reason instanceof Error ? reason.message : String(reason)}`);
  });
}

// ============================================================================
// Main Function
// ============================================================================

/**
 * Main function to run the server
 */
function main(): void {
  // Set up error handlers
  setupErrorHandlers();

  // Start the keep-alive mechanism
  keepAlive();

  // Set up readline interface
  const rl = createInterface({
    input: process.stdin,
    terminal: false
  });

  // Process each line from stdin
  rl.on('line', async (line: string) => {
    try {
      const truncatedLine = line.length > 500 ? `${line.substring(0, 500)}...` : line;
      log(`Received message: ${truncatedLine}`);
      
      const request = JSON.parse(line) as JsonRpcRequest;
      const { id, method, params } = request;
      
      log(`Processing method: ${method}, id: ${id}`);
      
      // Check if this is a notification (methods that start with 'notifications/')
      if (method.startsWith('notifications/')) {
        // Add specific handling for LLM response notifications
        if (method === 'notifications/llm_response') {
          logLlmResponse(params);
          return;
        }
        handleNotification(method, params);
        return;
      }
      
      // Handle different methods
      if (method === 'initialize') {
        handleInitialize(id, params);
      }
      else if (method === 'ListOfferings') {
        handleListOfferings(id);
      }
      else if (method === 'callTool') {
        // Store the image path before processing
        if (params?.arguments?.image_path) {
          conversationState.lastImagePath = params.arguments.image_path;
        }
        await handleCallTool(id, params);
      }
      else if (method === 'tools/list') {
        handleToolsList(id);
      }
      else if (method === 'tools/call') {
        // Store the image path before processing
        if (params?.image_path) {
          conversationState.lastImagePath = params.image_path;
        }
        await handleToolsCall(id, params);
      }
      else {
        log(`Method not supported: ${method}`);
        sendError(id, -32601, `Method not supported: ${method}`);
      }
    } catch (error) {
      log(`Error processing message: ${error instanceof Error ? error.message : String(error)}`);
      sendError(null, -32700, `Parse error or internal error: ${error instanceof Error ? error.message : String(error)}`);
    }
  });

  // Keep the process alive by not closing on stdin end
  rl.on('close', () => {
    log('stdin was closed, but keeping process alive');
    // Don't exit, just log
  });

  log('OpenAI OCR MCP Server ready and waiting for messages on stdin');
}

// ============================================================================
// Server Startup
// ============================================================================

// Start the server
main(); 

// Add handler for the new append_analysis tool
async function handleAppendAnalysis(args: { text_file_path: string; analysis: string }): Promise<McpToolResult> {
  try {
    const { text_file_path, analysis } = args;
    
    log(`Appending analysis to file: ${text_file_path}`);
    log(`Analysis content: ${analysis}`);

    // Verify the file exists and is a .txt file
    if (!fs.existsSync(text_file_path)) {
      throw new Error(`Text file not found: ${text_file_path}`);
    }
    if (!text_file_path.endsWith('.txt')) {
      throw new Error('File must be a .txt file');
    }

    // Append the analysis
    const timestamp = new Date().toISOString();
    const separator = '\n\nLLM ANALYSIS:\n=============\n';
    const appendText = `${separator}[${timestamp}]\n${analysis}\n`;

    fs.appendFileSync(text_file_path, appendText, 'utf8');
    log(`Successfully appended analysis to: ${text_file_path}`);

    return {
      content: [
        { type: "text", text: `Analysis has been appended to: ${text_file_path}` }
      ]
    };
  } catch (error) {
    log(`Error appending analysis: ${error instanceof Error ? error.message : String(error)}`);
    return {
      content: [
        { type: "text", text: `Error: ${error instanceof Error ? error.message : String(error)}` }
      ],
      isError: true
    };
  }
}

