import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser
from langchain_mistralai import ChatMistralAI

load_dotenv()

model = ChatMistralAI(model='mistral-small-2506')

class Movie(BaseModel):
    title: str
    release_year: Optional[int]
    genre: List[str]
    director: Optional[str]
    cast: List[str]
    rating: float
    summary: str

parser = PydanticOutputParser(pydantic_object=Movie)

prompt = ChatPromptTemplate.from_messages([
(
'system',
"""
Extract movie information from the paragraph
{format_instructions}
"""
),
('human', '{paragraph}')
])

st.title("Movie Information Extractor")

para = st.text_area("Give your paragraph:")

if st.button("Extract"):
  
    final_prompt = prompt.invoke(
       {
            "paragraph": para,
            "format_instructions": parser.get_format_instructions()
       }
    )

    response = model.invoke(final_prompt)

    st.write(response.content)

