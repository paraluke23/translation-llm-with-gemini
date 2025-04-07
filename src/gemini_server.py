# Standard Library Imports
import asyncio
import json
import logging
import os
import base64 # Keep if needed by future tools, currently unused

# Third-party Imports
from dotenv import load_dotenv
import nest_asyncio # Allows asyncio event loop nesting, sometimes needed

# Apply nest_asyncio early if running in environments like Jupyter/Colab
# or if MCP framework requires it internally.
nest_asyncio.apply()

# --- MCP Framework Import ---
# Import the specific MCP server class being used.
try:
    # Using FastMCP server framework from the 'mcp' library
    from mcp.server.fastmcp import Context, FastMCP
except ImportError:
    print("Error: Failed to import FastMCP from 'mcp.server.fastmcp'.")
    print("Please ensure the 'mcp' library is installed correctly and check its documentation.")
    exit(1)

# --- Google AI & Cloud Service Imports ---
try:
    # Google Generative AI (Gemini) library
    from google import genai
    from google.genai import types
    from google.api_core import exceptions as google_exceptions # For specific Google API error handling

    # Google Cloud Translation library
    from google.cloud import translate_v3 as translate

except ImportError:
    print("Error: 'google-generativeai' or 'google-cloud-translate' library not found.")
    print("Please install them: pip install google-generativeai google-cloud-translate google-cloud-aiplatform")
    exit(1)

# --- Logging Configuration ---
# Configure basic logging settings.
# Set level to ERROR to minimize noise, change to INFO or DEBUG for more verbosity.
logging.basicConfig(
    level=logging.ERROR, # Default level for the root logger
    format="%(asctime)s - %(levelname)s [%(name)s] - %(message)s" # Include logger name
)

# Suppress overly verbose logs from Google API libraries by setting their loggers to WARNING.
logging.getLogger('google.api_core').setLevel(logging.WARNING)
logging.getLogger('google.auth').setLevel(logging.WARNING)
logging.getLogger('google.generativeai').setLevel(logging.WARNING)
logging.getLogger('google.cloud').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)

# --- Environment & Configuration Loading ---
# Load environment variables from a .env file if it exists.
load_dotenv()

# Get Google Cloud configuration from environment variables.
# Provide default values, but it's better to set them in the .env file.
GOOGLE_PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
GOOGLE_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION")

# Validate that required configuration is present.
if not GOOGLE_PROJECT_ID or not GOOGLE_LOCATION:
    logging.error("FATAL: Environment variables GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION must be set.")
    # It's helpful to remind the user where to set them:
    logging.error("Please create a .env file or set them in your system environment.")
    exit(1)

# --- Initialize Google Clients ---
# Gemini Client (using Vertex AI backend)
genai_client = None # Initialize to None
try:
    # Configure the client to use Vertex AI, specifying project and location.
    genai_client = genai.Client(
        vertexai=True,
        project=GOOGLE_PROJECT_ID,
        location=GOOGLE_LOCATION,
    )
    # Use DEBUG level for successful initialization messages
    logging.debug(f"Gemini Client initialized via Vertex AI for project '{GOOGLE_PROJECT_ID}' in location '{GOOGLE_LOCATION}'")
except Exception as e:
    logging.error(f"Failed to initialize Gemini Client (Vertex AI): {e}")
    # Keep genai_client as None if initialization fails.

# Translation Client (initialized within the function for potentially better resource management)

# --- Instantiate MCP Server ---
mcp_host = None # Initialize to None
try:
    # Create an instance of the FastMCP server.
    # The name "gemini-complexity-server" should match the client's configuration.
    mcp_host = FastMCP("gemini-complexity-server")
    logging.debug(f"MCP Server '{mcp_host.name}' instantiated.")
except Exception as e:
    logging.error(f"Failed to instantiate FastMCP server: {e}")
    # Keep mcp_host as None if instantiation fails.


# --- Helper Functions ---

async def call_gemini_model(model_name: str, prompt: str) -> str:
    """
    Calls a specified Gemini model via the initialized Vertex AI client.

    Args:
        model_name: The full model identifier (e.g., "gemini-1.5-flash-latest").
        prompt: The text prompt to send to the model.

    Returns:
        The text response from the model.

    Raises:
        RuntimeError: If the Gemini client is not initialized or if an API call fails.
    """
    if not genai_client:
        logging.error("Attempted to call Gemini model, but client is not initialized.")
        raise RuntimeError("Gemini client not initialized.")

    logging.debug(f"Calling model '{model_name}' with prompt (first 70 chars): {prompt[:70]}...")

    # Prepare the content structure for the API call.
    contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]

    # Define generation parameters (temperature, safety settings, etc.).
    # Safety settings are turned OFF here - use with caution and ensure compliance
    # with responsible AI practices and Google's terms of service.
    # Consider adjusting these based on your use case.
    generate_content_config = types.GenerateContentConfig(
            temperature=0.2, # Lower temperature for more deterministic output
            top_p=0.8,       # Nucleus sampling parameter
            max_output_tokens=1024, # Limit response length
            # response_modalities=["TEXT"], # Often inferred, but can be explicit
            safety_settings=[
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"), # Changed to BLOCK_NONE
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),# Changed to BLOCK_NONE
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),# Changed to BLOCK_NONE
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE") # Changed to BLOCK_NONE
                # Valid thresholds: BLOCK_NONE, BLOCK_LOW_AND_ABOVE, BLOCK_MEDIUM_AND_ABOVE, BLOCK_ONLY_HIGH
            ]
        )

    try:
        # Make the API call using the genai client.
        response = genai_client.models.generate_content(
            model=model_name, # Pass the full model identifier
            contents=contents,
            config=generate_content_config,
        )

        # Process the response.
        if response and response.text:
            logging.debug(f"Received response from model '{model_name}'.")
            return response.text
        else:
            # Log a warning if the response structure is unexpected.
            logging.warning(f"Model '{model_name}' response did not contain text. Response: {response}")
            return "Error: Model returned an empty or unexpected response structure."

    except google_exceptions.GoogleAPIError as e:
        # Handle specific Google API errors.
        logging.error(f"Google API error calling model {model_name}: {e}")
        raise RuntimeError(f"Gemini API Error ({e.message or type(e).__name__})") from e
    except Exception as e:
        # Handle other unexpected errors during the API call.
        logging.exception(f"Unexpected error calling model {model_name}: {e}") # Use logging.exception to include traceback
        raise RuntimeError(f"Unexpected error in Gemini call ({type(e).__name__})") from e


def translate_text(
    project_id: str,
    location: str,
    source_language_code: str,
    target_language_code: str,
    source_text: str
) -> str | None:
    """
    Translates text using the Google Cloud Translation API.

    Args:
        project_id: Google Cloud project ID.
        location: Google Cloud location (e.g., 'us-central1', 'global').
        source_language_code: BCP-47 code of the source language (e.g., 'en', 'fr').
        target_language_code: BCP-47 code of the target language (e.g., 'fr', 'es').
        source_text: The text to translate.

    Returns:
        The translated text, or None if translation fails.
    """
    try:
        # Initialize the client within the function call.
        # Consider initializing once globally if called very frequently, but this is safer for resource management.
        client = translate.TranslationServiceClient()
        parent = f"projects/{project_id}/locations/{location}" # Use 'global' location for v3 API usually

        logging.debug(f"Calling Translation API: {source_language_code} -> {target_language_code} for text: {source_text[:50]}...")

        # Construct the translation request.
        request={
            "parent": parent,
            "contents": [source_text],
            "mime_type": "text/plain", # Assuming plain text input
            "source_language_code": source_language_code,
            "target_language_code": target_language_code,
        }

        # Make the API call.
        response = client.translate_text(request=request)

        # Process the response.
        if response.translations:
            translated = response.translations[0].translated_text
            logging.debug(f"Translation successful: {translated[:50]}...")
            return translated
        else:
            logging.warning("Translation API returned a response with no translations.")
            return None # Indicate failure clearly

    except google_exceptions.GoogleAPIError as e:
        logging.error(f"Google API error during translation: {e}")
        return None # Indicate failure clearly
    except Exception as e:
        logging.exception(f"Unexpected error during translation: {e}") # Log full traceback
        return None # Indicate failure clearly

# --- MCP Tool Definitions ---
# Use the @mcp_host.tool decorator to register functions as callable tools for MCP clients.
# The 'name' and 'description' are crucial for the client/LLM to understand and select the tool.

@mcp_host.tool(name="translate_llm", description="Translates text from a source language to a target language. Requires text, source_language (e.g., 'en'), and target_language (e.g., 'es').")
async def call_translate(text: str, source_language: str, target_language: str) -> str:
    """
    MCP Tool Function: Executes text translation using the Google Cloud Translation API.

    Args (provided by MCP client based on LLM):
        text: The text content to translate.
        source_language: The BCP-47 language code of the input text (e.g., "en", "auto" for detection).
        target_language: The BCP-47 language code for the desired output language (e.g., "fr", "de").

    Returns:
        The translated text, or an error message if translation fails.
    """
    # Note: GOOGLE_PROJECT_ID and GOOGLE_LOCATION are accessed from the global scope.
    logging.info(f"MCP Tool 'translate_llm' called: {source_language} -> {target_language}")
    try:
        # Call the helper function for translation.
        # Using 'global' as location is often correct for Translation API v3
        translated_result = translate_text(
            project_id=GOOGLE_PROJECT_ID,
            location='global', # Cloud Translate v3 often uses 'global'
            source_language_code=source_language,
            target_language_code=target_language,
            source_text=text
        )

        if translated_result is not None:
            return translated_result
        else:
            # Provide a meaningful error message back to the client/LLM.
            return "Error: Translation failed. Please check the language codes or try again later."
    except Exception as e:
        # Catch unexpected errors within the tool function itself.
        logging.exception(f"Unexpected error within MCP tool 'call_translate': {e}")
        return f"Error: An internal error occurred during translation: {e}"

# Tool for simpler, faster tasks, creative writing like poetry.
@mcp_host.tool(name="gemini_flash_lite_2_0", description="Calls a fast Gemini Flash Lite model. Good for simple Q&A, summaries, or creative tasks like writing poetry.")
async def call_gemini_flash_lite(prompt: str) -> str:
    """
    MCP Tool Function: Executes a prompt using the Gemini 2.0 Flash Lite model.

    Args:
        prompt: The user's prompt or query.

    Returns:
        The model's response text, or an error message.
    """
    # Define the specific model identifier. Check Vertex AI Model Garden for available models.
    model_name = "gemini-1.5-flash-latest" # Example: Use the latest flash model
    logging.info(f"MCP Tool 'gemini_flash_lite_2_0' called with model: {model_name}")
    try:
        return await call_gemini_model(model_name, prompt)
    except RuntimeError as e:
        return f"Error calling {model_name}: {e}" # Return error message to client
    except Exception as e:
        logging.exception(f"Unexpected error in MCP tool 'call_gemini_flash_lite': {e}")
        return f"Error: An internal error occurred calling {model_name}: {e}"

# Tool for tasks requiring some reasoning, like science explanations.
# NOTE: model name "gemini-2.0-flash-thinking-exp-01-21" might be experimental/internal.
# Replace with a generally available model if needed, e.g., gemini-2.5-pro-preview-03-25 might be better.
@mcp_host.tool(name="gemini_flash_thinking_2_0", description="Calls an experimental Gemini Flash model designed for reasoning tasks, like explaining scientific concepts.")
async def call_gemini_flash_thinking(prompt: str) -> str: # Renamed function
    """
    MCP Tool Function: Executes a prompt using an experimental Gemini Flash Thinking model.

    Args:
        prompt: The user's prompt or query, likely related to reasoning or science.

    Returns:
        The model's response text, or an error message.
    """
    # WARNING: This model name seems specific and potentially experimental/internal.
    # Verify its availability or replace with a suitable public model like 'gemini-2.5-pro-preview-03-25'.
    model_name = "gemini-2.5-pro-preview-03-25" # CHANGED to a more standard model
    logging.warning(f"MCP Tool 'gemini_flash_thinking_2_0' called. Using model: {model_name} (Original experimental name might not be available).")
    try:
        return await call_gemini_model(model_name, prompt)
    except RuntimeError as e:
        return f"Error calling {model_name}: {e}"
    except Exception as e:
        logging.exception(f"Unexpected error in MCP tool 'call_gemini_flash_thinking': {e}")
        return f"Error: An internal error occurred calling {model_name}: {e}"

# Tool for complex tasks: coding, math, multi-step reasoning.
# NOTE: model name "gemini-2.5-pro-exp-03-25" might be experimental/internal.
# Replace with a generally available powerful model like gemini-2.5-pro-preview-03-25.
@mcp_host.tool(name="gemini_pro_2_5", description="Calls a powerful Gemini Pro model for complex prompts involving coding, math, or multi-step reasoning.")
async def call_gemini_pro_2_5(prompt: str) -> str:
    """
    MCP Tool Function: Executes a prompt using a Gemini 1.5 Pro model (or similar).

    Args:
        prompt: The user's complex prompt (code, math, etc.).

    Returns:
        The model's response text, or an error message.
    """
    # WARNING: This model name seems specific and potentially experimental/internal.
    # Verify its availability or replace with a suitable public model like 'gemini-2.5-pro-preview-03-25'.
    model_name = "gemini-2.5-pro-preview-03-25" # CHANGED to the standard latest Pro model
    logging.warning(f"MCP Tool 'gemini_pro_2_5' called. Using model: {model_name} (Original experimental name might not be available).")
    try:
        return await call_gemini_model(model_name, prompt)
    except RuntimeError as e:
        return f"Error calling {model_name}: {e}"
    except Exception as e:
        logging.exception(f"Unexpected error in MCP tool 'call_gemini_pro_2_5': {e}")
        return f"Error: An internal error occurred calling {model_name}: {e}"

# --- Main Execution ---
def main():
    """
    Sets up and runs the MCP server.
    This function is the entry point when the script is executed directly.
    """
    # Perform checks before starting the server.
    if not genai_client:
        logging.error("Cannot start server: Gemini client failed to initialize.")
        return # Exit if client setup failed
    if not mcp_host:
        logging.error("Cannot start server: MCP Host (FastMCP) failed to instantiate.")
        return # Exit if server setup failed

    # Log server startup information.
    # The transport mechanism (stdio) is typically handled by the client starting this script.
    logging.info(f"Starting MCP server '{mcp_host.name}'...")
    logging.info("Server ready to accept connections (typically via stdio from MCP client)...")

    try:
        # Run the MCP server's main loop.
        # This will likely block until the server is terminated.
        # FastMCP's run() method handles the underlying event loop and communication.
        mcp_host.run() # No transport argument needed here usually

    except Exception as e:
        # Log any fatal error during server execution.
        logging.exception("MCP Host run() method failed unexpectedly:")
        print(f"Error running MCP Host: {e}") # Also print to stderr for visibility
    finally:
        logging.info(f"MCP server '{mcp_host.name}' shutting down.")

# Standard Python entry point check.
if __name__ == "__main__":
    # Execute the main function to start the server.
    main()
