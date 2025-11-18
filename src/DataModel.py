from pydantic import BaseModel, Field
from typing import Dict

class MHQuestionAnswers(BaseModel):
    QuestionAnswers: list[Dict[str, str]] = Field(description="list of dictionaries. Each dictionary has the three keys; 'Question' , 'Answer' and 'Answer Explanation'")
