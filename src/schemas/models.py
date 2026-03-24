from typing import Any, Dict, List, Optional, Literal

from pydantic import BaseModel, Field


class DraftRequest(BaseModel):
    raw_notes: str = Field(..., min_length=1)
    user_answers: str = Field("", description="Answers to clarifying questions (optional)")
    purpose: str = Field(..., min_length=1)
    audience: str = Field(..., min_length=1)
    tone: str = Field(..., min_length=1)
    language: str = Field("English", min_length=1)
    channel: str = Field("Email", min_length=1)
    include_subject: bool = True
    word_size: Literal["Small", "Medium", "Large"] = "Medium"
    draft_variants: int = Field(3, ge=1, le=5)
    # Used for draft selection scoring (tone/wording preferences).
    style_preset: str = Field("Professional", min_length=1)
    finalize_requested: bool = False
    seed: Optional[int] = None

    # Ollama generation controls (passed to /api/generate options).
    # Keep defaults so the app works without the user touching advanced settings.
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    top_k: int = Field(40, ge=0, le=200)

    repeat_penalty: float = Field(1.1, ge=0.0, le=2.5)
    presence_penalty: float = Field(0.0, ge=-2.0, le=2.5)
    frequency_penalty: float = Field(0.0, ge=-2.0, le=2.5)


class DraftResponse(BaseModel):
    questions: List[str] = []
    draft: str = ""
    rubric_check: Dict[str, Any] = Field(default_factory=dict)
    final: str = ""

