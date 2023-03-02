from pydantic import BaseModel


class TextData(BaseModel):
    """This class defines the data that is expected in the request body when
    calling the /predict endpoint."""

    text: str
