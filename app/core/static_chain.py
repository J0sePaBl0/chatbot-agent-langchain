"""Example of a static linear chain — no LangGraph, no tool-calling loop.

Steps (always executed in this order, no decisions made):
  1. Summarize the raw document text in 2-3 sentences
  2. Extract the 3 most important key points from that summary
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.config import settings


def build_summarizer_chain():
    llm = ChatOpenAI(
        model=settings.CHAT_MODEL,
        temperature=0,
        api_key=settings.OPENAI_API_KEY,
    )

    summarize_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize the following document text in 2-3 sentences."),
        ("human", "{text}"),
    ])

    key_points_prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract the 3 most important key points from this summary as a numbered list."),
        ("human", "{text}"),
    ])

    chain = (
        summarize_prompt
        | llm
        | StrOutputParser()
        | (lambda summary: {"text": summary})
        | key_points_prompt
        | llm
        | StrOutputParser()
    )

    return chain


async def summarize_document(text: str) -> str:
    chain = build_summarizer_chain()
    return await chain.ainvoke({"text": text})
