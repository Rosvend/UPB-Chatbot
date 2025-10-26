"""
Embeddings Module
Handles creation and management of embeddings for document chunks.
"""

import os
from typing import Optional
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings

load_dotenv()


def get_embeddings(provider: str = "azure") -> AzureOpenAIEmbeddings | OpenAIEmbeddings:
    """
    Initialize embeddings model based on provider.
    
    Args:
        provider: "azure" or "openai"
    
    Returns:
        Embeddings instance
    
    Raises:
        ValueError: If required environment variables are missing
    """
    if provider == "azure":
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
        
        if not api_key or not endpoint:
            raise ValueError(
                "Missing Azure OpenAI credentials. Set AZURE_OPENAI_API_KEY and "
                "AZURE_OPENAI_ENDPOINT in your .env file"
            )
        
        return AzureOpenAIEmbeddings(
            azure_deployment=deployment,
            openai_api_version="2024-02-01",
            azure_endpoint=endpoint,
            api_key=api_key,
        )
    
    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY in your .env file")
        
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=api_key,
        )
    
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'azure' or 'openai'")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    print("Testing embeddings initialization...\n")
    
    # Try Azure first
    try:
        print("Attempting Azure OpenAI...")
        embeddings = get_embeddings("azure")
        print("Azure OpenAI embeddings initialized")
        
        # Test with a sample text
        test_text = "Ingeniería de Sistemas en la UPB"
        vector = embeddings.embed_query(test_text)
        print(f"Embedding dimension: {len(vector)}")
        print(f"Sample values: {vector[:5]}")
        
    except ValueError as e:
        print(f"Azure setup failed: {e}\n")
        
        # Fallback to OpenAI
        try:
            print("Attempting OpenAI...")
            embeddings = get_embeddings("openai")
            print("OpenAI embeddings initialized")
            
            test_text = "Ingeniería de Sistemas en la UPB"
            vector = embeddings.embed_query(test_text)
            print(f"Embedding dimension: {len(vector)}")
            print(f"Sample values: {vector[:5]}")
            
        except ValueError as e:
            print(f"OpenAI setup failed: {e}")
            print("\nPlease configure your .env file with credentials")
