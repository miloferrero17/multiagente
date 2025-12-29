#!/usr/bin/env python3
"""
Robust wrapper lazy-loading the OpenAI client.
This file will NOT raise at import time if 'openai' is missing; it raises when ask_openai is invoked.
"""
import os, json
from typing import Any, List, Dict, Union

def _get_client():
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("Missing dependency 'openai'. Install it in your environment: pip install openai") from e
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in env/.env.")
    return OpenAI(api_key=api_key)

def _resp_to_text(response: Any) -> str:
    try:
        if hasattr(response, "output_text") and response.output_text:
            return response.output_text
    except Exception:
        pass
    try:
        return json.dumps(response, default=str, ensure_ascii=False)
    except Exception:
        return str(response)

def ask_openai(messages: Union[str, List[Dict[str,str]]], temperature: float = 0.0, model: str = "gpt-4.1") -> str:
    client = _get_client()
    try:
        resp = client.responses.create(model=model, input=messages, temperature=temperature)
        return _resp_to_text(resp)
    except Exception as e:
        raise RuntimeError(f"Error calling OpenAI API: {e}") from e

