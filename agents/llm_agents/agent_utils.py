''' Utils specific to agents '''

from __future__ import annotations

from pydantic import BaseModel, computed_field

class LactChainAgentMessage(BaseModel):
    ''' Container class for displaying agent messages based on success or failure '''
    success: bool
    num_tries: int
    task: str
    code: str

    @computed_field
    @property
    def message(self) -> str:
        if self.success:
            return f'Agent successfully generated code for task {self.task} in {self.num_tries} tries!'
        else:
            return f'Agent unsuccessfully generated code for task {self.task} in {self.num_tries} :('