"""Public API contracts. Algorithm parameters stay in the existing pipelines."""
from typing import Literal, Any
from pydantic import BaseModel, ConfigDict, Field, model_validator

class Contract(BaseModel):
    model_config = ConfigDict(extra='forbid')

class RunRequest(Contract):
    workflow: Literal['manipulation', 'locomotion', 'export']
    recording_id: str = Field(min_length=1, max_length=200, pattern=r'^[\w.:-]+$')
    robot_id: str = Field(min_length=1, max_length=100, pattern=r'^[\w.-]+$')
    hand_id: str | None = Field(default=None, max_length=100, pattern=r'^[\w.-]+$')

    @model_validator(mode='after')
    def explicit_selection(self):
        if self.workflow == 'manipulation' and self.hand_id is None:
            raise ValueError('Select hand_id explicitly, including "none"')
        if self.workflow != 'manipulation' and self.hand_id is not None:
            raise ValueError('hand_id applies only to manipulation')
        return self

class BatchRequest(Contract):
    items: list[RunRequest] = Field(min_length=1, max_length=100)

class JobView(Contract):
    id: str
    workflow: str
    status: str
    created_at: float
    updated_at: float
    cache_hit: bool = False
    backend_job_id: str | None = None
    result: dict[str, Any] = Field(default_factory=dict)
    error: dict[str, Any] | None = None
    status_url: str
    events_url: str
    physics_status: str = 'not_run'

class PlanView(Contract):
    executable: bool
    request: RunRequest
    operation: str
    stages: list[str]
    policy: dict[str, Any]
    limitations: list[str]
