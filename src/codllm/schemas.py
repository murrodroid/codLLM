from typing import List,Literal,Optional
from pydantic import BaseModel,Field


class SingleCOD(BaseModel):
    Letter: Literal['A','B'] = Field()

class ICD10h:
    CausesOfDeaths: List[SingleCOD]