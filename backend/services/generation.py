import logging 
from typing import List, Dict, Any , Literal
from dataclasses import dataclass
from enum import Enum

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate 
from langchain_groq import ChatGroq 


logger = logging.getLogger(__name__)

class ResponseFormat(str, Enum): 
    NARRATIVE = "narrative"
    STRUCTURED = "structured"
    CONCISE = "concise"
    RESEARCH = "research"

@dataclass 
class GenerationConfig: 
    """Configuration for answer generation behvaiour"""
    system_message : str| None = None 
    response_format: ResponseFormat = ResponseFormat.NARRATIVE
    temperature: float = 0.3
    max_tokens: int = 512 
    include_reasoning: bool = False 
    citation_style: Literal["inline","footnote","numbered"] = "inline"


class GroqGenerator: 

    def __init__(self, 
                 model_name: str = "openai/gpt-oss-20b",
                **kwargs
                ):
        

        logger.info(f"Loading Groq model: {model_name}")
        self.llm = ChatGroq(
            model_name = model_name,
            **kwargs
        )
        logger.info(f"Groq model loaded: {model_name}")

        self.system_presets = {
            "legal": "You are an expert legal analyst. Focus on applicable laws, precedents, and nuanced interpretations. Be precise and thorough.",
            "technical": "You are a technical expert. Explain concepts clearly with practical examples. Focus on implementation details and trade-offs.",
            "researcher": "You are a research synthesizer. Integrate multiple perspectives, highlight key findings, and identify gaps or contradictions.",
            "simple": "Answer clearly and concisely. Use simple language. Focus on the most important points.",
        }

    def _format_context(self, 
                    documents: List[Dict[str, Any]], 
                    citation_style: Literal["inline", "footnote", "numbered"] = "inline"
                ) -> str: 
        """Format documents into context string with citation style."""
        context_parts = []

        for i, doc in enumerate(documents, 1): 
            # Handle both dict and Document objects
            if isinstance(doc, dict):
                source = doc.get('metadata', {}).get('source', 'unknown')
                page = doc.get('metadata', {}).get('page', 'unknown')
                content = doc.get('content', '')
                chunk_id = doc.get('chunk_id', '')
            else:
                # Fallback for LangChain Document objects
                source = doc.metadata.get('source', 'unknown')
                page = doc.metadata.get('page', 'unknown')
                content = doc.page_content
                chunk_id = ''

            # Format based on citation style
            if citation_style == "inline":
                # [1] content (Source: file.pdf, Page: 5)
                context_parts.append(
                    f"[{i}] {content}\n"
                    f"    (Source: {source}, Page: {page})"
                )
            
            elif citation_style == "footnote":
                # content [1]
                context_parts.append(
                    f"{content} [{i}]\n"
                    f"Footnote {i}: Source: {source}, Page: {page}"
                )
            
            elif citation_style == "numbered":
                # content
                context_parts.append(content)
                if i == len(documents):
                    # Add all sources at the end
                    sources_info = "\n".join(
                        [f"{j}. Source: {d.get('metadata', {}).get('source', 'unknown')}, Page: {d.get('metadata', {}).get('page', 'unknown')}" 
                         for j, d in enumerate(documents, 1)]
                    )

        return "\n\n".join(context_parts)

    def _get_system_message(self, 
                            config: GenerationConfig
                        ) -> str: 
        """Determine system message based on config."""
        if config.system_message: 
            return config.system_message 

        format_defaults = {
            ResponseFormat.NARRATIVE: "Provide a comprehensive answer with citations.",
            ResponseFormat.STRUCTURED: "Organize your answer with clear sections and bullet points.",
            ResponseFormat.CONCISE: "Provide a brief, direct answer with essential sources.",
            ResponseFormat.RESEARCH: "Synthesize information from multiple sources. Show analysis and reasoning.",
        }
        return format_defaults.get(config.response_format, format_defaults[ResponseFormat.NARRATIVE])
    
    def _build_prompt(self, query: str, documents: List[Dict], config: GenerationConfig) -> str:
        """Build prompt dynamically based on config."""
        context = self._format_context(documents, config.citation_style)
        
        if config.include_reasoning:
            answer_instruction = "First, explain your reasoning. Then provide the answer."
        else:
            answer_instruction = "Provide the answer."
        
        prompt = f"""{self._get_system_message(config)}

            Context:
            {context}

            Question: {query}

            {answer_instruction}

            Answer:"""
        return prompt
    

    def generate(self, 
                 query: str, 
                 documents: List[Document], 
                 config: GenerationConfig  | None = None
                ) -> str: 
        """Generate answer with felxible configuration"""

        if config is None: 
            config = GenerationConfig()

        prompt = self._build_prompt(query, documents, config)
        response = self.llm.invoke(
            prompt, 
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )

        return response.content.strip()
        


    def generate_with_metadata(self, 
                               query: str, 
                               documents: List[Document],
                               config: GenerationConfig | None = None
                            )-> Dict[str,Any]: 
        """Generate answer and return with source metadata."""
        if config is None: 
            config = GenerationConfig()

        answer = self.generate(query, documents, config)

        sources = [
            {
                "source": doc['metadata'].get('source', 'unknown'),
                "page": doc['metadata'].get('page', 'unknown'),
                "rerank_score": doc['metadata'].get('rerank_score', None),
                "preview": doc['content'][:100]+"..." if len(doc['content'])>100 else doc['content']  
            }
            for doc in documents
        ]

        return {
            "answer": answer,
            "sources": sources,
            "query": query,
            "num_sources": len(sources),
            "response_format": config.response_format.value,
            "temperature": config.temperature,
        }
    

    def get_system_preset(self, name:str) -> str: 
        return self.system_presets.get(name)
    
    def set_system_preset(self, name:str, message:str): 
        self.system_presets[name] = message