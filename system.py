"""
OKNLAB AI Platform - Core System Architecture
Enterprise-grade multi-agent orchestration with live RAG
"""

import os
import asyncio
import logging
from typing import Dict, List, Any, Optional, Type
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field
from langchain.llms.base import LLM
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langgraph.graph import StateGraph, END
from langchain.schema import Document

import requests
from bs4 import BeautifulSoup
import chromadb
from chromadb.config import Settings

# ========================================
# LOGGING CONFIGURATION
# ========================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("OKNLAB")


# ========================================
# CONFIGURATION MANAGEMENT
# ========================================

class Config:
    """Centralized configuration"""
    
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
    MODEL_SERVER = os.getenv("MODEL_SERVER", "http://localhost:8000/v1")
    API_KEY = os.getenv("API_KEY", "oknlab-local-key")
    
    GOOGLE_CSE_API_KEY = os.getenv("GOOGLE_CSE_API_KEY", "")
    GOOGLE_CSE_CX = os.getenv("GOOGLE_CSE_CX", "")
    
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
    
    VECTOR_DB_PATH = "./data/chroma_db"
    
    DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
    DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "2048"))


# ========================================
# CUSTOM LLM WRAPPER FOR QWEN
# ========================================

class QwenLLM(LLM):
    """Custom LLM wrapper for Qwen model via vLLM server"""
    
    model_name: str = Config.MODEL_NAME
    api_base: str = Config.MODEL_SERVER
    api_key: str = Config.API_KEY
    temperature: float = Config.DEFAULT_TEMPERATURE
    max_tokens: int = Config.DEFAULT_MAX_TOKENS
    
    @property
    def _llm_type(self) -> str:
        return "qwen"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Execute LLM call to vLLM server"""
        try:
            response = requests.post(
                f"{self.api_base}/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "stop": stop or []
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json()["choices"][0]["text"].strip()
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"ERROR: {str(e)}"


# ========================================
# LIVE WEB SCRAPING RAG ENGINE
# ========================================

class LiveRAGEngine:
    """Real-time RAG with Google CSE web scraping"""
    
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=Config.VECTOR_DB_PATH
        ))
        
        self.vectorstore = None
        logger.info("LiveRAGEngine initialized")
    
    def search_web(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        """Search web using Google CSE"""
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": Config.GOOGLE_CSE_API_KEY,
                "cx": Config.GOOGLE_CSE_CX,
                "q": query,
                "num": num_results
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            results = []
            for item in response.json().get("items", []):
                results.append({
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "snippet": item.get("snippet", "")
                })
            
            logger.info(f"Found {len(results)} results for: {query}")
            return results
        
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return []
    
    def scrape_content(self, url: str) -> str:
        """Scrape content from URL"""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove scripts and styles
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text[:10000]  # Limit to 10k chars
        
        except Exception as e:
            logger.error(f"Scraping failed for {url}: {e}")
            return ""
    
    def build_knowledge_base(self, query: str) -> bool:
        """Build vector store from live web scraping"""
        try:
            # Search web
            search_results = self.search_web(query)
            if not search_results:
                logger.warning("No search results found")
                return False
            
            # Scrape and process content
            documents = []
            for result in search_results:
                content = self.scrape_content(result["link"])
                if content:
                    documents.append(Document(
                        page_content=content,
                        metadata={
                            "source": result["link"],
                            "title": result["title"]
                        }
                    ))
            
            if not documents:
                logger.warning("No content scraped")
                return False
            
            # Split and embed
            splits = self.text_splitter.split_documents(documents)
            
            # Create vector store
            self.vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory=Config.VECTOR_DB_PATH
            )
            
            logger.info(f"Knowledge base built: {len(splits)} chunks from {len(documents)} sources")
            return True
        
        except Exception as e:
            logger.error(f"Knowledge base build failed: {e}")
            return False
    
    def query(self, question: str, llm: LLM) -> str:
        """Query the knowledge base"""
        if not self.vectorstore:
            return "Knowledge base not initialized. Run build_knowledge_base first."
        
        try:
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3})
            )
            
            result = qa_chain.run(question)
            return result
        
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            return f"ERROR: {str(e)}"


# ========================================
# AGENT FRAMEWORK
# ========================================

class AgentType(str, Enum):
    """Agent type enumeration"""
    CODE_ARCHITECT = "code_architect"
    SEC_ANALYST = "sec_analyst"
    AUTO_BOT = "auto_bot"
    AGENT_SUITE = "agent_suite"
    CREATIVE = "creative"
    CUSTOM = "custom"


class AgentConfig(BaseModel):
    """Agent configuration model"""
    name: str
    type: AgentType
    description: str
    temperature: float = 0.7
    max_tokens: int = 2048
    system_prompt: str = ""
    tools: List[str] = Field(default_factory=list)
    rag_enabled: bool = False


class BaseAgent(ABC):
    """Abstract base agent class"""
    
    def __init__(self, config: AgentConfig, llm: LLM, rag_engine: Optional[LiveRAGEngine] = None):
        self.config = config
        self.llm = llm
        self.rag_engine = rag_engine
        self.execution_count = 0
        self.logger = logging.getLogger(f"Agent.{config.name}")
    
    @abstractmethod
    async def execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent task"""
        pass
    
    def _build_prompt(self, task: str, context: Dict[str, Any]) -> str:
        """Build prompt with system prompt and context"""
        prompt_parts = []
        
        if self.config.system_prompt:
            prompt_parts.append(f"SYSTEM: {self.config.system_prompt}")
        
        if context:
            prompt_parts.append(f"CONTEXT: {context}")
        
        prompt_parts.append(f"TASK: {task}")
        
        return "\n\n".join(prompt_parts)
    
    async def _execute_with_rag(self, task: str) -> str:
        """Execute with RAG if enabled"""
        if self.config.rag_enabled and self.rag_engine:
            # Build knowledge base from task
            await asyncio.to_thread(self.rag_engine.build_knowledge_base, task)
            # Query with RAG
            return await asyncio.to_thread(self.rag_engine.query, task, self.llm)
        else:
            # Direct LLM call
            prompt = self._build_prompt(task, {})
            return await asyncio.to_thread(self.llm, prompt)


# ========================================
# SPECIALIZED AGENTS
# ========================================

class CodeArchitectAgent(BaseAgent):
    """Code generation and architecture agent"""
    
    async def execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        self.execution_count += 1
        self.logger.info(f"Executing code task: {task[:50]}...")
        
        # Enhanced system prompt for coding
        self.config.system_prompt = """You are an expert software architect and developer.
        Generate production-ready, secure, and optimized code.
        Include error handling, logging, and documentation.
        Support: Python, JavaScript, Rust, Go, TypeScript."""
        
        result = await self._execute_with_rag(task)
        
        return {
            "agent": self.config.name,
            "type": "code",
            "task": task,
            "result": result,
            "timestamp": datetime.now().isoformat(),
            "execution_count": self.execution_count
        }


class SecAnalystAgent(BaseAgent):
    """Security analysis and penetration testing agent"""
    
    async def execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        self.execution_count += 1
        self.logger.info(f"Executing security task: {task[:50]}...")
        
        self.config.system_prompt = """You are a cybersecurity expert specializing in:
        - Penetration testing
        - Vulnerability assessment
        - Threat modeling
        - Security audits
        - OWASP Top 10
        Provide actionable security recommendations."""
        
        result = await self._execute_with_rag(task)
        
        return {
            "agent": self.config.name,
            "type": "security",
            "task": task,
            "result": result,
            "vulnerabilities": self._extract_vulnerabilities(result),
            "timestamp": datetime.now().isoformat()
        }
    
    def _extract_vulnerabilities(self, result: str) -> List[str]:
        """Extract vulnerability mentions"""
        vuln_keywords = ["XSS", "SQL injection", "CSRF", "vulnerability", "exploit"]
        return [kw for kw in vuln_keywords if kw.lower() in result.lower()]


class AutoBotAgent(BaseAgent):
    """Automation and workflow orchestration agent"""
    
    async def execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        self.execution_count += 1
        self.logger.info(f"Executing automation task: {task[:50]}...")
        
        self.config.system_prompt = """You are an automation specialist.
        Design and implement workflows using:
        - Apache Airflow DAGs
        - API integrations (REST, GraphQL)
        - Zapier/n8n/Make workflows
        - Event-driven architectures
        Provide executable automation scripts."""
        
        result = await self._execute_with_rag(task)
        
        return {
            "agent": self.config.name,
            "type": "automation",
            "task": task,
            "result": result,
            "workflow_generated": True,
            "timestamp": datetime.now().isoformat()
        }


class AgentSuiteAgent(BaseAgent):
    """Admin, finance, and operations automation agent"""
    
    async def execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        self.execution_count += 1
        self.logger.info(f"Executing admin task: {task[:50]}...")
        
        self.config.system_prompt = """You are an operations and admin specialist.
        Handle:
        - Report generation
        - Financial analysis
        - Data processing (CSV, Excel)
        - Meeting notes and summaries
        - Documentation automation"""
        
        result = await self._execute_with_rag(task)
        
        return {
            "agent": self.config.name,
            "type": "admin",
            "task": task,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }


class CreativeAgent(BaseAgent):
    """Content creation agent (text, audio, visual)"""
    
    async def execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        self.execution_count += 1
        self.logger.info(f"Executing creative task: {task[:50]}...")
        
        self.config.system_prompt = """You are a creative content specialist.
        Generate:
        - Marketing copy
        - Articles and blog posts
        - Social media content
        - Documentation
        - Creative writing
        SEO-optimized, engaging, and professional."""
        
        result = await self._execute_with_rag(task)
        
        return {
            "agent": self.config.name,
            "type": "creative",
            "task": task,
            "result": result,
            "content_type": self._detect_content_type(task),
            "timestamp": datetime.now().isoformat()
        }
    
    def _detect_content_type(self, task: str) -> str:
        """Detect content type from task"""
        task_lower = task.lower()
        if "article" in task_lower or "blog" in task_lower:
            return "article"
        elif "social" in task_lower or "tweet" in task_lower:
            return "social"
        elif "marketing" in task_lower or "copy" in task_lower:
            return "marketing"
        return "general"


class CustomAgent(BaseAgent):
    """User-defined custom agent"""
    
    async def execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        self.execution_count += 1
        self.logger.info(f"Executing custom task: {task[:50]}...")
        
        result = await self._execute_with_rag(task)
        
        return {
            "agent": self.config.name,
            "type": "custom",
            "task": task,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }


# ========================================
# AGENT ORCHESTRATOR
# ========================================

class AgentOrchestrator:
    """Central orchestrator for multi-agent system"""
    
    def __init__(self):
        self.llm = QwenLLM()
        self.rag_engine = LiveRAGEngine()
        self.agents: Dict[str, BaseAgent] = {}
        self.execution_history: List[Dict[str, Any]] = []
        
        # Agent type mapping
        self.agent_classes: Dict[AgentType, Type[BaseAgent]] = {
            AgentType.CODE_ARCHITECT: CodeArchitectAgent,
            AgentType.SEC_ANALYST: SecAnalystAgent,
            AgentType.AUTO_BOT: AutoBotAgent,
            AgentType.AGENT_SUITE: AgentSuiteAgent,
            AgentType.CREATIVE: CreativeAgent,
            AgentType.CUSTOM: CustomAgent
        }
        
        self._initialize_default_agents()
        logger.info("AgentOrchestrator initialized")
    
    def _initialize_default_agents(self):
        """Initialize default agent suite"""
        default_configs = [
            AgentConfig(
                name="CodeArchitect",
                type=AgentType.CODE_ARCHITECT,
                description="Expert code generation and architecture",
                rag_enabled=True
            ),
            AgentConfig(
                name="SecAnalyst",
                type=AgentType.SEC_ANALYST,
                description="Security analysis and penetration testing",
                rag_enabled=True
            ),
            AgentConfig(
                name="AutoBot",
                type=AgentType.AUTO_BOT,
                description="Workflow automation and API orchestration",
                rag_enabled=True
            ),
            AgentConfig(
                name="AgentSuite",
                type=AgentType.AGENT_SUITE,
                description="Admin, finance, and operations automation",
                rag_enabled=False
            ),
            AgentConfig(
                name="CreativeAgent",
                type=AgentType.CREATIVE,
                description="Content creation and marketing",
                rag_enabled=True
            )
        ]
        
        for config in default_configs:
            self.create_agent(config)
    
    def create_agent(self, config: AgentConfig) -> BaseAgent:
        """Create and register new agent"""
        agent_class = self.agent_classes.get(config.type, CustomAgent)
        agent = agent_class(config, self.llm, self.rag_engine)
        self.agents[config.name] = agent
        logger.info(f"Agent created: {config.name} ({config.type})")
        return agent
    
    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get agent by name"""
        return self.agents.get(name)
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """List all registered agents"""
        return [
            {
                "name": agent.config.name,
                "type": agent.config.type.value,
                "description": agent.config.description,
                "execution_count": agent.execution_count,
                "rag_enabled": agent.config.rag_enabled
            }
            for agent in self.agents.values()
        ]
    
    async def execute_task(self, agent_name: str, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute task with specified agent"""
        agent = self.get_agent(agent_name)
        if not agent:
            raise ValueError(f"Agent not found: {agent_name}")
        
        result = await agent.execute(task, context or {})
        self.execution_history.append(result)
        
        return result
    
    async def execute_multi_agent(self, tasks: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Execute multiple agents in parallel"""
        coroutines = [
            self.execute_task(task["agent"], task["task"], task.get("context"))
            for task in tasks
        ]
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        return results
    
    def get_execution_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get execution history"""
        return self.execution_history[-limit:]
    
    def delete_agent(self, name: str) -> bool:
        """Delete custom agent"""
        if name in self.agents:
            del self.agents[name]
            logger.info(f"Agent deleted: {name}")
            return True
        return False


# ========================================
# WORKFLOW GRAPH (LANGGRAPH)
# ========================================

class WorkflowState(BaseModel):
    """Workflow state for LangGraph"""
    task: str
    agent_sequence: List[str]
    current_index: int = 0
    results: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None


class WorkflowOrchestrator:
    """LangGraph-based workflow orchestration"""
    
    def __init__(self, agent_orchestrator: AgentOrchestrator):
        self.orchestrator = agent_orchestrator
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build workflow graph"""
        workflow = StateGraph(WorkflowState)
        
        # Define nodes
        workflow.add_node("execute_agent", self._execute_agent_node)
        workflow.add_node("check_completion", self._check_completion_node)
        
        # Define edges
        workflow.set_entry_point("execute_agent")
        workflow.add_edge("execute_agent", "check_completion")
        workflow.add_conditional_edges(
            "check_completion",
            self._should_continue,
            {
                "continue": "execute_agent",
                "end": END
            }
        )
        
        return workflow.compile()
    
    async def _execute_agent_node(self, state: WorkflowState) -> WorkflowState:
        """Execute current agent in sequence"""
        if state.current_index >= len(state.agent_sequence):
            return state
        
        agent_name = state.agent_sequence[state.current_index]
        
        try:
            result = await self.orchestrator.execute_task(agent_name, state.task)
            state.results.append(result)
            state.current_index += 1
        except Exception as e:
            state.error = str(e)
            logger.error(f"Workflow error: {e}")
        
        return state
    
    def _check_completion_node(self, state: WorkflowState) -> WorkflowState:
        """Check if workflow is complete"""
        return state
    
    def _should_continue(self, state: WorkflowState) -> str:
        """Determine if workflow should continue"""
        if state.error or state.current_index >= len(state.agent_sequence):
            return "end"
        return "continue"
    
    async def execute_workflow(self, task: str, agent_sequence: List[str]) -> Dict[str, Any]:
        """Execute multi-agent workflow"""
        initial_state = WorkflowState(
            task=task,
            agent_sequence=agent_sequence
        )
        
        final_state = await self.graph.ainvoke(initial_state)
        
        return {
            "task": task,
            "agents": agent_sequence,
            "results": final_state["results"],
            "error": final_state.get("error"),
            "completed": final_state["current_index"] == len(agent_sequence)
        }


# ========================================
# EXPORT SYSTEM COMPONENTS
# ========================================

__all__ = [
    "Config",
    "QwenLLM",
    "LiveRAGEngine",
    "AgentType",
    "AgentConfig",
    "BaseAgent",
    "CodeArchitectAgent",
    "SecAnalystAgent",
    "AutoBotAgent",
    "AgentSuiteAgent",
    "CreativeAgent",
    "CustomAgent",
    "AgentOrchestrator",
    "WorkflowOrchestrator"
]
