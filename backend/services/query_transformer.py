"""HyDE and Multi-Query implementation for query transformation."""

import logging 
from typing import List, Dict, Any

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate 

logger = logging.getLogger(__name__)

class QueryTransformer: 
    
    def __init__(self, llm_model: str = "openai/gpt-oss-20b"):
        self.llm = ChatGroq(
            model_name=llm_model,
            temperature=0.7,
            max_tokens=512,
        )

    def hyde(self, query:str) -> str: 
        """Generate a HyDE-style synthetic document from the query."""
        prompt = PromptTemplate(
            input_variables=["query"],
            template="""Write a detailed answer to the following question. 
            The answer should be comprehensive and specific.
            
        Question: {query}

        Hypothetical Answer:"""
        )

        try:
            result = self.llm.invoke(prompt.format(query=query))
            answer = result.content
            logger.debug(f"Generated HyDE answer ({len(answer)} chars)")
            return answer
        except Exception as e:
            logger.error(f"Error generating HyDE: {e}")
            return ""
        

    def multi_query(self, query:str, num_queries: int = 3) -> List[str]:
        """Generate multiple diverse query phrasings"""
        prompt = PromptTemplate(
            input_variables = ["query", "num"],
            template="""Generate {num} alternative ways to phrase the following question.
            Each should be a complete question and explore different aspects of the original.
            Return as a numbered list.

            Original Question: {query}

            Alternative Phrasings:"""
            )
        
        try: 
            result = self.llm.invoke(prompt.format(query=query, num=num_queries))
            content = result.content

            lines = content.split('\n')
            queries = []
            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith("-")):
                    q = line.lstrip("0123456789.- ").strip()
                    if q:
                        queries.append(q)

            return queries[:num_queries]

        except Exception as e:
            logger.error(f"Error generating multi-queries: {e}")
            return []
        
    def transform(self, query:str, use_hyde: bool = False,use_multi: bool = False, num_multi_queries: int = 3) -> Dict[str, Any]:
        queries = []

        if use_multi: 
            alternatives = self.multi_query(query, num_multi_queries)
            queries.extend(alternatives)

        if use_hyde:
            hyde_doc = self.hyde(query)
            if hyde_doc:
                queries.append(hyde_doc)

        return queries

        
