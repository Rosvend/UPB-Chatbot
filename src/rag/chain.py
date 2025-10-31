"""
RAG Chain with Conversation Memory
Implements conversational RAG with GPT-4o-mini and source citations.
"""

import os
from pathlib import Path
import sys
from typing import List, Dict
from operator import itemgetter

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document


class UPBRAGChain:
    """
    Conversational RAG chain for UPB career exploration.
    Features: GPT-4o-mini, conversation memory, source citations.
    """
    
    def __init__(self, retriever, retrieval_method: str = "hybrid"):
        """
        Initialize RAG chain with retriever.
        
        Args:
            retriever: UPBRetriever instance
            retrieval_method: Method for retrieval (bm25, similarity, mmr, hybrid)
        """
        self.retriever = retriever
        self.retrieval_method = retrieval_method
        self.chat_history = []
        
        # Initialize LLM
        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_LLM_DEPLOYMENT", "gpt-4o-mini"),
            openai_api_version="2024-02-01",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0.7,
        )
        
        # Create prompt template with conversation memory
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Eres un asistente virtual de la Universidad Pontificia Bolivariana (UPB) especializado en orientación académica. 
Tu rol es ayudar a estudiantes prospecto a explorar y comprender los programas de ingeniería ofrecidos por la UPB.

REGLAS CRÍTICAS PARA PREVENIR CONFUSIÓN DE INFORMACIÓN:
1. Cada fragmento de contexto tiene etiquetas [PROGRAMA:], [CODIGO:], [SECCION:] que identifican el programa específico
2. NUNCA mezcles información entre programas diferentes
3. Si preguntan sobre un programa específico (ej: Ingeniería de Sistemas), usa SOLO fragmentos con [PROGRAMA: Ingeniería de Sistemas e Informática]
4. Si el plan de estudios/pensum está en el contexto de un programa, esa información es SOLO para ese programa
5. Verifica siempre las etiquetas de contexto antes de afirmar que algo existe en un programa

Características de tus respuestas:
- Usa un tono amigable, cercano y profesional
- Responde en español de manera clara y concisa
- Basa tus respuestas ÚNICAMENTE en el contexto proporcionado
- Si no encuentras información relevante en el contexto, di "No tengo información sobre eso."
- Menciona EXPLÍCITAMENTE el nombre del programa cuando respondas (ej: "En Ingeniería Industrial sí se ve Cálculo Vectorial, pero en Ingeniería de Sistemas no")
- Si es apropiado, sugiere programas relacionados que puedan interesar al estudiante

Contexto relevante (cada fragmento pertenece a UN programa específico):
{context}"""),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}")
        ])
        
        # Build the RAG chain
        self.chain = (
            RunnablePassthrough.assign(
                context=itemgetter("question") | RunnableLambda(self._retrieve_and_format)
            )
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _retrieve_and_format(self, question: str) -> str:
        """
        Retrieve relevant documents and format them as context.
        Each document is clearly separated and labeled to prevent confusion.
        
        Args:
            question: User question
            
        Returns:
            Formatted context string with clear boundaries
        """
        docs = self.retriever.retrieve(
            question, 
            method=self.retrieval_method, 
            k=6
        )
        
        self.last_retrieved_docs = docs
        
        formatted_docs = []
        for i, doc in enumerate(docs, 1):
            content = doc.page_content.strip()
            
            formatted_docs.append(
                f"--- INICIO FRAGMENTO {i} ---\n{content}\n--- FIN FRAGMENTO {i} ---"
            )
        
        return "\n\n".join(formatted_docs)
    
    def invoke(self, question: str, include_sources: bool = True) -> Dict:
        """
        Invoke the RAG chain with a question.
        
        Args:
            question: User question
            include_sources: Whether to include source citations in response
            
        Returns:
            Dict with 'answer' and optionally 'sources'
        """
        # Invoke chain with question and chat history
        answer = self.chain.invoke({
            "question": question,
            "chat_history": self.chat_history
        })
        
        # Update chat history
        self.chat_history.extend([
            HumanMessage(content=question),
            AIMessage(content=answer)
        ])
        
        # Prepare response
        response = {"answer": answer}
        
        if include_sources and hasattr(self, 'last_retrieved_docs'):
            sources = []
            for doc in self.last_retrieved_docs:
                source_info = {
                    "content": doc.page_content[:200] + "...",
                    "category": doc.metadata.get('category', 'N/A'),
                    "source": doc.metadata.get('source', 'N/A')
                }
                sources.append(source_info)
            response["sources"] = sources
        
        return response
    
    def clear_history(self):
        """Clear conversation history."""
        self.chat_history = []
    
    def get_history_summary(self) -> str:
        """Get a summary of conversation history."""
        if not self.chat_history:
            return "No hay historial de conversación."
        
        summary = []
        for i, msg in enumerate(self.chat_history):
            role = "Usuario" if isinstance(msg, HumanMessage) else "Asistente"
            content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            summary.append(f"{i+1}. {role}: {content}")
        
        return "\n".join(summary)


if __name__ == "__main__":
    from setup_retrieval import setup_retrieval_system
    
    print("=" * 70)
    print("UPB RAG CHAIN - CONVERSATIONAL TEST")
    print("=" * 70)
    
    # Setup retrieval system
    print("\nInitializing retrieval system...")
    retriever, vectorstore_manager, chunks = setup_retrieval_system(
        vectorstore_path="vectorstore/faiss_index",
        use_existing=True
    )
    
    # Create RAG chain
    print("\nCreating RAG chain with GPT-4o-mini...")
    rag_chain = UPBRAGChain(retriever, retrieval_method="hybrid")
    print("RAG chain ready!")
    
    # Test conversation flow
    print("\n" + "=" * 70)
    print("CONVERSATION TEST")
    print("=" * 70)
    
    # Question 1
    question1 = "¿Se ve cálculo vectorial en la ingeniería de sistemas en la UPB?"
    print(f"\nUsuario: {question1}")
    print("-" * 70)
    
    response1 = rag_chain.invoke(question1, include_sources=True)
    print(f"Asistente: {response1['answer']}")
    
    if 'sources' in response1:
        print("\n[Fuentes utilizadas]")
        for i, source in enumerate(response1['sources'], 1):
            print(f"{i}. Categoría: {source['category']}")
            print(f"   Archivo: {source['source']}")
            print(f"   Contenido: {source['content']}\n")
    
    # Question 2 (with context from previous question)
    print("\n" + "=" * 70)
    question2 = "¿Se ve ecuaciones diferenciales en ingeniería en diseño y entretenimiento digital en la UPB?"
    print(f"Usuario: {question2}")
    print("-" * 70)
    
    response2 = rag_chain.invoke(question2, include_sources=True)
    print(f"Asistente: {response2['answer']}")
    
    if 'sources' in response2:
        print("\n[Fuentes utilizadas]")
        for i, source in enumerate(response2['sources'], 1):
            print(f"{i}. Categoría: {source['category']}")
            print(f"   Archivo: {source['source']}")
    
    # Question 3 (memory test)
    print("\n" + "=" * 70)
    question3 = "¿Cuánto cuesta el semestre de esa carrera?"
    print(f"Usuario: {question3}")
    print("-" * 70)
    
    response3 = rag_chain.invoke(question3, include_sources=True)
    print(f"Asistente: {response3['answer']}")
    
    # Show conversation history
    print("\n" + "=" * 70)
    print("HISTORIAL DE CONVERSACIÓN")
    print("=" * 70)
    print(rag_chain.get_history_summary())
    
    print("\n" + "=" * 70)
    print("RAG CHAIN TEST COMPLETE")
    print("=" * 70)
    print("\nFeatures tested:")
    print("- GPT-4o-mini integration")
    print("- Hybrid retrieval (BM25 + Vector with RRF)")
    print("- Conversation memory")
    print("- Source citations")
    print("- Multi-turn dialogue")
