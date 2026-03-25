"""Pydantic v2 models for the Code Review Environment."""

from pydantic import BaseModel, Field


class Observation(BaseModel):
    """What the agent sees at each step."""

    code_snippet: str = Field(description="Python code snippet to review")
    task_id: str = Field(description="Current task identifier")
    context: str = Field(description="Instructions or hints for the current step")
    step_number: int = Field(ge=0, description="Current step in the episode")


class Action(BaseModel):
    """What the agent submits as a response."""

    response: str = Field(description="Agent's answer: yes/no, bug type, or fixed code")
    task_id: str = Field(description="Task identifier this action is for")


class Reward(BaseModel):
    """Grading result returned to the agent."""

    score: float = Field(ge=0.0, le=1.0, description="Reward score in [0.0, 1.0]")
    feedback: str = Field(default="", description="Human-readable grading feedback")
