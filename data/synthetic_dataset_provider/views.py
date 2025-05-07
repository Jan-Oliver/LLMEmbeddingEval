from pydantic import BaseModel, Field

class EmailMetadata(BaseModel):
    """
    Pydantic model representing metadata for a synthetic email.
    """
    industry: str = Field(..., description="Industry domain of the email sender/recipient")
    emotion: str = Field(..., description="Primary emotion of the email")
