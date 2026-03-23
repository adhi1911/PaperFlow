import logging 
from typing import List, Dict, Any , Optional 

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate 
from langchain_groq import ChatGroq 


logger = logging.getLogger(__name__)

class GroqGenerator: 

    def __init__(self, 
                 model_name: str = "openai/gpt-oss-20b",
                 temperature: float = 0.3,
                 max_tokens: int = 512
                ):
        

        logger.info(f"Loading Groq model: {model_name}")
        self.llm = ChatGroq(
            model_name = model_name,
            temperature = temperature,
            max_tokens = max_tokens
        )
        logger.info(f"Groq model loaded: {model_name}")

    def _format_context(self, 
                        documents: List[Document]
                        
                    ) -> str: 
        context_parts = []

        for i , doc in enumerate(documents,1): 
            source = doc['metadata'].get('source', 'unknown')
            page = doc['metadata'].get('page', 'unknown')
            content = doc['content']

            context_parts.append(
                f"[{i}] {content}\n"
                f"(Source: {source}, Page: {page})\n"
            )

        return "\n".join(context_parts)
    

    def generate(self, 
                query: str, 
                documents: List[Document],
                include_citations: bool = True,
                ) -> str: 
        
        if not documents:
            return "No relevant information found in the documents."
        
        context = self._format_context(documents)
        if include_citations: 
            instruction = (
                "Answer using ONLY the information from the provided context. "
                "Cite sources in brackets like [1], [2], etc. corresponding to the context sections."
            )

        else:
            instruction = "Answer the questions using the provided context" 
        
        prompt_template = ChatPromptTemplate.from_template( f"""{instruction}
                                                    Context:
                                                    {{context}}

                                                    Question: {{question}}

                                                    Answer:"""
                                                    )
        try: 
            response = self.llm.invoke(
                prompt_template.format_messages(context = context, question = query)
            )

            answer = response.content
            return answer 
        
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            return "An error occurred while generating the answer."
        


    def generate_with_metadata(self, 
                               query: str, 
                               documents: List[Dict[str,Any]],
                            )-> Dict[str,Any]: 

        answer = self.generate(query, documents)

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
            "num_sources": len(sources)
        }