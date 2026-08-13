"""
LLM generation module.
Provider-agnostic wrapper supporting Gemini, OpenAI, and Groq
with RAG prompting, query rewriting, and streaming.
"""

import os
from typing import AsyncIterator, Optional

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.language_models.chat_models import BaseChatModel

from backend.config import get_settings


# --- Prompt Templates ---

RAG_SYSTEM_PROMPT = """You are an AI Document Intelligence Assistant. Your role is to answer questions based ONLY on the provided document context.

Rules:
1. Answer the question using ONLY the information found in the context below.
2. If the answer is not in the context, clearly state: "I couldn't find this information in the uploaded documents."
3. Always cite which document and page number supports your answer using the format: [Document Name, Page X].
4. Be precise, clear, and well-structured in your response.
5. If multiple sources are relevant, reference all of them.
6. Do not make up information or use knowledge outside the provided context."""

RAG_USER_TEMPLATE = """Context from uploaded documents:

{context}

---

Question: {question}

Provide a detailed answer based on the context above, with citations."""

QUERY_REWRITE_PROMPT = """You are a helpful assistant that rewrites user questions to be more effective for document search.

Given the conversation history and the latest question, rewrite the question to:
1. Resolve any pronouns or references (e.g., "it", "this", "that") using conversation context.
2. Make the question self-contained and specific.
3. Keep the rewritten question concise.

Only output the rewritten question, nothing else.

Conversation history:
{history}

Latest question: {question}

Rewritten question:"""

SUMMARIZE_PROMPT = """Summarize the following document content in a structured format.
Include these sections if applicable:
• **Main Objective / Purpose**
• **Key Points / Methodology**
• **Results / Findings**
• **Limitations**
• **Conclusions / Future Work**

Be thorough but concise. Use bullet points for clarity.

Document content:
{context}"""

MCQ_PROMPT = """Generate {count} multiple-choice questions (MCQs) from the following document content.

For each question:
1. Provide 4 options (A, B, C, D)
2. Mark the correct answer
3. Provide a brief explanation

Format each question clearly with numbers.

Document content:
{context}"""

VIVA_PROMPT = """Generate {count} viva/oral examination questions from the following document content.

Questions should:
1. Range from basic to advanced difficulty
2. Test understanding, not just memorization
3. Include expected key points in the answer

Format: Number each question and provide expected answer points.

Document content:
{context}"""

EXPLAIN_PROMPT = """Explain the following topic in simple, easy-to-understand language.
Use analogies and examples where helpful. Assume the reader is a student learning this for the first time.

Topic/Content:
{context}

Question: {question}"""

TOPICS_PROMPT = """Identify the most important topics from the following document content.
For each topic:
1. Topic name
2. Why it's important
3. Key points to remember

Rank topics by importance.

Document content:
{context}"""


def get_llm(
    provider: Optional[str] = None,
    temperature: float = 0.3,
    streaming: bool = False,
) -> BaseChatModel:
    """
    Factory function to get the appropriate LLM based on provider.

    Args:
        provider: LLM provider name ("gemini", "openai", "groq").
                  Defaults to the value in settings.
        temperature: LLM temperature (0.0 = deterministic, 1.0 = creative).
        streaming: Whether to enable streaming mode.

    Returns:
        A LangChain chat model instance.
    """
    settings = get_settings()
    provider = provider or settings.llm_provider

    if provider == "gemini":
        os.environ["GOOGLE_API_KEY"] = settings.google_api_key
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=temperature,
            streaming=streaming,
        )

    elif provider == "openai":
        os.environ["OPENAI_API_KEY"] = settings.openai_api_key
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=temperature,
            streaming=streaming,
        )

    elif provider == "groq":
        os.environ["GROQ_API_KEY"] = settings.groq_api_key
        from langchain_groq import ChatGroq
        return ChatGroq(
            model="llama-3.1-70b-versatile",
            temperature=temperature,
            streaming=streaming,
        )

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}. Use 'gemini', 'openai', or 'groq'.")


async def generate_answer(
    question: str,
    context: str,
    conversation_history: Optional[list[dict]] = None,
) -> str:
    """
    Generate a grounded answer using the RAG pipeline.

    Args:
        question: The user's question.
        context: Retrieved context from documents.
        conversation_history: Previous messages for multi-turn context.

    Returns:
        The LLM's answer as a string.
    """
    llm = get_llm()

    messages = [SystemMessage(content=RAG_SYSTEM_PROMPT)]

    # Add conversation history if available
    if conversation_history:
        for msg in conversation_history[-6:]:  # Last 6 messages for context
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

    # Add the RAG-augmented question
    user_message = RAG_USER_TEMPLATE.format(context=context, question=question)
    messages.append(HumanMessage(content=user_message))

    response = await llm.ainvoke(messages)
    return response.content


async def generate_answer_stream(
    question: str,
    context: str,
    conversation_history: Optional[list[dict]] = None,
) -> AsyncIterator[str]:
    """
    Stream a grounded answer token by token.

    Yields:
        String tokens as they are generated.
    """
    llm = get_llm(streaming=True)

    messages = [SystemMessage(content=RAG_SYSTEM_PROMPT)]

    if conversation_history:
        for msg in conversation_history[-6:]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

    user_message = RAG_USER_TEMPLATE.format(context=context, question=question)
    messages.append(HumanMessage(content=user_message))

    async for chunk in llm.astream(messages):
        if chunk.content:
            yield chunk.content


async def rewrite_query(
    question: str,
    conversation_history: Optional[list[dict]] = None,
) -> str:
    """
    Rewrite an ambiguous query using conversation history.

    Args:
        question: The user's latest question.
        conversation_history: Previous messages for context.

    Returns:
        The rewritten, self-contained question.
    """
    if not conversation_history:
        return question

    llm = get_llm(temperature=0.1)

    # Format history
    history_text = ""
    for msg in conversation_history[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    prompt = QUERY_REWRITE_PROMPT.format(history=history_text, question=question)
    messages = [HumanMessage(content=prompt)]

    response = await llm.ainvoke(messages)
    rewritten = response.content.strip()

    return rewritten if rewritten else question


async def generate_student_content(
    mode: str,
    context: str,
    question: Optional[str] = None,
    count: int = 10,
) -> str:
    """
    Generate student-focused content (summaries, MCQs, viva questions, etc.).

    Args:
        mode: One of "summarize", "mcqs", "viva", "explain", "topics".
        context: Document context to work with.
        question: Optional specific question (for "explain" mode).
        count: Number of items to generate (for MCQs, viva questions).

    Returns:
        Generated content as a string.
    """
    llm = get_llm(temperature=0.5)

    prompt_map = {
        "summarize": SUMMARIZE_PROMPT.format(context=context),
        "mcqs": MCQ_PROMPT.format(context=context, count=count),
        "viva": VIVA_PROMPT.format(context=context, count=count),
        "explain": EXPLAIN_PROMPT.format(context=context, question=question or "Explain this content"),
        "topics": TOPICS_PROMPT.format(context=context),
    }

    prompt = prompt_map.get(mode)
    if prompt is None:
        raise ValueError(f"Unknown student mode: {mode}. Use: {list(prompt_map.keys())}")

    messages = [HumanMessage(content=prompt)]
    response = await llm.ainvoke(messages)
    return response.content
