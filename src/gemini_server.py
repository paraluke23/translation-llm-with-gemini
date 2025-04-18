# gemini_server.py

import nest_asyncio
nest_asyncio.apply()
import json
import asyncio
import logging
import os
from dotenv import load_dotenv

# --- Hypothetical High-Level MCP Import ---
try:
    # Using mcp server framework
    from mcp.server.fastmcp import Context, FastMCP
except ImportError:
    print("Error: Failed to import a high-level MCP server class (tried 'mcp.host.MCPHost').")
    print("Please check the documentation for your 'mcp' library version.")
    exit(1)


# --- Import Google GenAI ---
try:
    # import google.generativeai as genai <- Use specific import below
    from google import genai
    from google.genai import types
    import base64
    from google.api_core import exceptions as google_exceptions # For more specific error handling
    from google.cloud import translate_v3 as translate # Import the Cloud Translation library

except ImportError:
    print("Error: 'google-generativeai' or 'google-cloud-translate' library not found.")
    print("Please install them: pip install google-generativeai google-cloud-translate")
    exit(1)

# Configure logging - Set default level higher (WARNING or ERROR)
logging.basicConfig(
    level=logging.ERROR, # Set to WARNING to hide INFO messages
    format="%(asctime)s - %(levelname)s [%(name)s] - %(message)s" # Added logger name
)

# --- Suppress Verbose Google API Logs ---
# Set levels for specific noisy loggers
logging.getLogger('google.api_core').setLevel(logging.WARNING)
logging.getLogger('google.auth').setLevel(logging.WARNING)
logging.getLogger('google.generativeai').setLevel(logging.WARNING)
logging.getLogger('google.cloud').setLevel(logging.WARNING) # Add others if needed
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)

 # --- Configuration ---
GOOGLE_PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "cloud-ml-translation-test")
# Use the location from the example or environment variable
GOOGLE_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

if not GOOGLE_PROJECT_ID or not GOOGLE_LOCATION:
    logging.error("Environment variables GOOGLE_PROJECT_ID and GOOGLE_LOCATION must be set.")
    exit(1)


# --- Initialize Gemini Client ---
try:
    genai_client = genai.Client(
        vertexai=True,
        project=GOOGLE_PROJECT_ID,
        location=GOOGLE_LOCATION,
    )
    logging.info(f"Gemini Client initialized for project '{GOOGLE_PROJECT_ID}' in location '{GOOGLE_LOCATION}'") # Keep this INFO? Maybe change to debug
except Exception as e:
    logging.error(f"Failed to initialize Gemini Client: {e}")
    genai_client = None

# --- Instantiate High-Level MCP Server ---
try:
    mcp_host = FastMCP("gemini-complexity-server")
except NameError: # Handle case where MCPHost import failed silently
    logging.error("MCPHost class not available. Cannot create MCP server.")
    exit(1)
except Exception as e:
    logging.error(f"Failed to instantiate MCPHost: {e}")
    exit(1)


# --- Common Gemini API Call Function ---
async def call_gemini_model(model_name: str, prompt: str) -> str:
    """Calls the specified Gemini model using the google-genai library."""
    if not genai_client:
        raise RuntimeError("Gemini client not initialized.")

    # Changed to debug level
    logging.debug(f"Calling model '{model_name}' for prompt: {prompt[:70]}...")
    contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]

    generate_content_config = types.GenerateContentConfig(
        temperature = 0.2,
        top_p = 0.8,
        max_output_tokens = 1024,
        response_modalities = ["TEXT"],
        safety_settings = [types.SafetySetting(
            category="HARM_CATEGORY_HATE_SPEECH",
            threshold="OFF"
        ),types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold="OFF"
        ),types.SafetySetting(
            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
            threshold="OFF"
        ),types.SafetySetting(
            category="HARM_CATEGORY_HARASSMENT",
            threshold="OFF"
        )]
    )

    try:
        response = genai_client.models.generate_content(
            model=model_name, # Pass the model ID/path
            contents=contents,
            config=generate_content_config, # Pass the config object
        )
        if response:
            return response.text
        else:
            logging.warning(f"Model '{model_name}' response candidate has no text parts.") # Keep warnings
            return "Error: Model returned a response structure without text content."

    except google_exceptions.GoogleAPIError as e:
        logging.error(f"Google API error calling model {model_name}: {e}") # Keep errors
        raise RuntimeError(f"Gemini API Error ({e.message or type(e).__name__})") from e
    except Exception as e:
        logging.exception(f"Unexpected error calling model {model_name}: {e}") # Keep errors/exceptions
        raise RuntimeError(f"Unexpected error in Gemini call ({type(e).__name__})") from e


def translate_text(
    project_id: str,
    location: str,
    source_language_code: str,
    target_language_code: str,
    source_text: str
) -> str | None:
    """Translates text using the Google Cloud Translation API."""
    try:
        client = translate.TranslationServiceClient()
        parent = f"projects/{project_id}/locations/{location}"

        response = client.translate_text(
            request={
                "parent": parent,
                "contents": [source_text],
                "mime_type": "text/plain",
                "source_language_code": source_language_code,
                "target_language_code": target_language_code,
            }
        )
        if response.translations:
            return response.translations[0].translated_text
        else:
            logging.warning("Warning: No translations found in the response.")
            return None
    except Exception as e:
        logging.exception(f"Translation API call failed: {e}")
        return None

# # --- Tool Definitions using Decorator ---
@mcp_host.tool(name="translate_llm", description="Calls this translate_llm tool for exclusively for requests explicitly asking for language translation or meaning clarification of non-English text.")
async def call_translate(text: str, source_language: str, target_language: str) -> str:
    """Executes a prompt using the Translation API."""
    try:
        PROJECT_ID = GOOGLE_PROJECT_ID
        LOCATION = GOOGLE_LOCATION  # Or your specific region
        SOURCE_LANG = source_language
        TARGET_LANG = target_language
        TEXT_TO_TRANSLATE = text

        translated_result = translate_text(
            project_id=PROJECT_ID,
            location=LOCATION,
            source_language_code=SOURCE_LANG,
            target_language_code=TARGET_LANG,
            source_text=TEXT_TO_TRANSLATE
        )

        if translated_result:
            return translated_result
        else:
            return "\nTranslation failed."
    except Exception as e:
        logging.exception("MCP Host run failed:")
        print(f"Error running MCP Host: {e}")

@mcp_host.tool(name="gemini_flash_lite_2_0", description="Calls the Gemini 2.0 Flash Lite model for poetry prompts.")
async def call_gemini_pro(prompt: str) -> str:
    """Executes a prompt using the Gemini Pro model."""
    model_name = "gemini-2.0-flash-lite-001"
    return await call_gemini_model(model_name, prompt)

@mcp_host.tool(name="gemini_flash_thinking_2_0", description="Calls the Gemini 2.0 Flash Thinking model for prompts relating to science.")
async def call_gemini_pro(prompt: str) -> str:
    """Executes a prompt using the Gemini Pro model."""
    model_name = "gemini-2.0-flash-thinking-exp-01-21"
    return await call_gemini_model(model_name, prompt)

@mcp_host.tool(name="gemini_pro_2_5", description="Calls the Gemini 2.5 Pro Thinking model for complex prompts, code prompts, math prompts where thinking is needed.")
async def call_gemini_pro_2_5(prompt: str) -> str:
    """Executes a prompt using the Gemini 1.5 Pro model."""
    model_name = "gemini-2.5-pro-exp-03-25"
    return await call_gemini_model(model_name, prompt)

# --- Main Execution Function (Now Synchronous) ---
def main():
    """Sets up and runs the MCP server using the high-level host."""
    if not genai_client:
        logging.error("Cannot start server: Gemini client failed to initialize.")
        return
    if 'mcp_host' not in globals():
        logging.error("Cannot start server: MCPHost failed to instantiate.")
        return

    logging.info(f"Starting MCP server '{mcp_host.name}' with stdio transport...")
    try:
        # Call run() directly - it will start its own event loop (likely using anyio)
        # Remove the explicit transport='stdio' argument unless required by your specific MCPHost class
        mcp_host.run()
    except Exception as e:
        logging.exception("MCP Host run failed:")
        print(f"Error running MCP Host: {e}")


if __name__ == "__main__":
    # Call the synchronous main function directly
    main()
