"""
OKNLAB AI Platform - Main Orchestrator
FastAPI server with vLLM, multi-agent system, and live RAG
"""

import os
import asyncio
import logging
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dotenv import load_dotenv
from pyngrok import ngrok

# Import system components
from system import (
    Config,
    AgentOrchestrator,
    WorkflowOrchestrator,
    AgentConfig,
    AgentType
)

# Load environment
load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OKNLAB")

# ========================================
# GLOBAL STATE
# ========================================

orchestrator: Optional[AgentOrchestrator] = None
workflow_orchestrator: Optional[WorkflowOrchestrator] = None
websocket_clients: List[WebSocket] = []


# ========================================
# API MODELS
# ========================================

class TaskRequest(BaseModel):
    agent_name: str
    task: str
    context: Optional[Dict[str, Any]] = None
    rag_query: Optional[str] = None


class MultiTaskRequest(BaseModel):
    tasks: List[Dict[str, str]]


class WorkflowRequest(BaseModel):
    task: str
    agent_sequence: List[str]


class AgentCreateRequest(BaseModel):
    name: str
    type: str
    description: str
    temperature: float = 0.7
    max_tokens: int = 2048
    system_prompt: str = ""
    rag_enabled: bool = False


class RAGQueryRequest(BaseModel):
    query: str
    web_search: bool = True


# ========================================
# FASTAPI APP
# ========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic"""
    global orchestrator, workflow_orchestrator
    
    # Startup
    logger.info("🚀 Starting OKNLAB AI Platform...")
    
    # Initialize orchestrator
    orchestrator = AgentOrchestrator()
    workflow_orchestrator = WorkflowOrchestrator(orchestrator)
    
    # Setup NGROK tunnel
    if os.getenv("NGROK_AUTH_TOKEN"):
        ngrok.set_auth_token(os.getenv("NGROK_AUTH_TOKEN"))
        public_url = ngrok.connect(8080, bind_tls=True)
        logger.info(f"🌐 Public URL: {public_url}")
    
    logger.info("✅ OKNLAB Platform Ready")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down OKNLAB Platform")
    ngrok.disconnect(public_url.public_url if 'public_url' in locals() else None)


app = FastAPI(
    title="OKNLAB AI Platform",
    description="Enterprise Multi-Agent Orchestration System",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========================================
# WEBSOCKET FOR REAL-TIME UPDATES
# ========================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time agent updates"""
    await websocket.accept()
    websocket_clients.append(websocket)
    logger.info("WebSocket client connected")
    
    try:
        while True:
            data = await websocket.receive_text()
            # Echo for now - can implement real-time task streaming
            await websocket.send_json({"status": "connected", "message": data})
    except WebSocketDisconnect:
        websocket_clients.remove(websocket)
        logger.info("WebSocket client disconnected")


async def broadcast_update(data: Dict[str, Any]):
    """Broadcast update to all connected clients"""
    for client in websocket_clients:
        try:
            await client.send_json(data)
        except:
            pass


# ========================================
# API ENDPOINTS
# ========================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve dashboard HTML"""
    try:
        with open("OKNLAB — Dashboard.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Dashboard not found. Please ensure OKNLAB — Dashboard.html exists.</h1>")


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": Config.MODEL_NAME,
        "agents": len(orchestrator.agents) if orchestrator else 0,
        "rag_enabled": True
    }


@app.get("/api/agents")
async def list_agents():
    """List all registered agents"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    return {
        "agents": orchestrator.list_agents(),
        "total": len(orchestrator.agents)
    }


@app.post("/api/agents/create")
async def create_agent(request: AgentCreateRequest):
    """Create new custom agent"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    try:
        agent_type = AgentType(request.type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid agent type: {request.type}")
    
    config = AgentConfig(
        name=request.name,
        type=agent_type,
        description=request.description,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        system_prompt=request.system_prompt,
        rag_enabled=request.rag_enabled
    )
    
    agent = orchestrator.create_agent(config)
    
    await broadcast_update({
        "event": "agent_created",
        "agent": request.name
    })
    
    return {
        "status": "success",
        "agent": {
            "name": agent.config.name,
            "type": agent.config.type.value,
            "description": agent.config.description
        }
    }


@app.delete("/api/agents/{agent_name}")
async def delete_agent(agent_name: str):
    """Delete custom agent"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    success = orchestrator.delete_agent(agent_name)
    
    if not success:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_name}")
    
    await broadcast_update({
        "event": "agent_deleted",
        "agent": agent_name
    })
    
    return {"status": "success", "deleted": agent_name}


@app.post("/api/execute")
async def execute_task(request: TaskRequest, background_tasks: BackgroundTasks):
    """Execute single agent task"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    try:
        # Optional: Build RAG knowledge base first
        if request.rag_query:
            rag_query = request.rag_query or request.task
            background_tasks.add_task(
                orchestrator.rag_engine.build_knowledge_base,
                rag_query
            )
        
        result = await orchestrator.execute_task(
            request.agent_name,
            request.task,
            request.context
        )
        
        await broadcast_update({
            "event": "task_completed",
            "agent": request.agent_name,
            "task": request.task[:50]
        })
        
        return result
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Task execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/execute/multi")
async def execute_multi_task(request: MultiTaskRequest):
    """Execute multiple agents in parallel"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    try:
        results = await orchestrator.execute_multi_agent(request.tasks)
        
        return {
            "status": "success",
            "results": results,
            "total": len(results)
        }
    
    except Exception as e:
        logger.error(f"Multi-task execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/workflow")
async def execute_workflow(request: WorkflowRequest):
    """Execute multi-agent workflow with LangGraph"""
    if not workflow_orchestrator:
        raise HTTPException(status_code=503, detail="Workflow orchestrator not initialized")
    
    try:
        result = await workflow_orchestrator.execute_workflow(
            request.task,
            request.agent_sequence
        )
        
        await broadcast_update({
            "event": "workflow_completed",
            "task": request.task[:50],
            "agents": request.agent_sequence
        })
        
        return result
    
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rag/query")
async def rag_query(request: RAGQueryRequest):
    """Query RAG engine with live web search"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    try:
        if request.web_search:
            # Build knowledge base from web
            success = await asyncio.to_thread(
                orchestrator.rag_engine.build_knowledge_base,
                request.query
            )
            
            if not success:
                raise HTTPException(status_code=500, detail="Failed to build knowledge base")
        
        # Query with LLM
        answer = await asyncio.to_thread(
            orchestrator.rag_engine.query,
            request.query,
            orchestrator.llm
        )
        
        return {
            "query": request.query,
            "answer": answer,
            "web_search": request.web_search
        }
    
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/history")
async def get_history(limit: int = 100):
    """Get execution history"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    return {
        "history": orchestrator.get_execution_history(limit),
        "total": len(orchestrator.execution_history)
    }


@app.get("/api/stats")
async def get_statistics():
    """Get system statistics"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    total_executions = sum(agent.execution_count for agent in orchestrator.agents.values())
    
    return {
        "total_agents": len(orchestrator.agents),
        "total_executions": total_executions,
        "execution_history": len(orchestrator.execution_history),
        "agents_breakdown": [
            {
                "name": agent.config.name,
                "type": agent.config.type.value,
                "executions": agent.execution_count
            }
            for agent in orchestrator.agents.values()
        ]
    }


# ========================================
# MAIN EXECUTION
# ========================================

if __name__ == "__main__":
    import uvicorn
    
    # vLLM server should be started separately:
    # python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-1.5B-Instruct --port 8000
    
    logger.info("="*50)
    logger.info("OKNLAB AI PLATFORM")
    logger.info("="*50)
    logger.info(f"Model: {Config.MODEL_NAME}")
    logger.info(f"Model Server: {Config.MODEL_SERVER}")
    logger.info(f"RAG Enabled: True (Live Web Scraping)")
    logger.info("="*50)
    
    uvicorn.run(
        "run_colab:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info"
    )
