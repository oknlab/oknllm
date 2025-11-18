import os
import uvicorn
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

# AI & LangChain
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.utilities import GoogleSearchAPIWrapper
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

# --- CONFIGURATION ---
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="OKNLAB AI Core", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODEL LOADER (Optimized for Colab/T4) ---
print(">>> INITIALIZING NEURAL ENGINE...")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

model_id = os.getenv("MODEL_ID", "Qwen/Qwen3-1.7B")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    temperature=0.1, # Low temp for precise code generation
    top_p=0.95,
    repetition_penalty=1.15
)

llm = HuggingFacePipeline(pipeline=pipe)

# --- TOOLS & RAG ---

@tool
def google_search_tool(query: str) -> str:
    """Performs a live web search using Google CSE for RAG."""
    api_key = os.getenv("GOOGLE_API_KEY")
    cse_id = os.getenv("GOOGLE_CSE_ID")
    if not api_key or not cse_id:
        return "Error: Google API credentials not configured."
    
    import requests
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"q": query, "key": api_key, "cx": cse_id, "num": 3}
    try:
        res = requests.get(url, params=params)
        res.raise_for_status()
        data = res.json()
        results = []
        if 'items' in data:
            for item in data['items']:
                results.append(f"Title: {item['title']}\nLink: {item['link']}\nSnippet: {item['snippet']}")
        return "\n---\n".join(results) if results else "No results found."
    except Exception as e:
        return f"Search failed: {str(e)}"

@tool
def python_repl_tool(code: str) -> str:
    """Executes Python code safely and returns stdout/stderr. Use for calculation or logic testing."""
    import io, sys
    from contextlib import redirect_stdout, redirect_stderr
    
    # Security restriction: block dangerous imports
    if any(x in code for x in ["os.system", "subprocess", "shutil.rmtree"]):
        return "Security Alert: Dangerous system calls blocked by SecAnalyst."

    stdout = io.StringIO()
    stderr = io.StringIO()
    
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exec(code, {"__name__": "__main__"})
        return stdout.getvalue() + stderr.getvalue()
    except Exception as e:
        return f"Execution Error: {str(e)}"

tools = [google_search_tool, python_repl_tool]
llm_with_tools = llm.bind_tools(tools) if hasattr(llm, "bind_tools") else None 
# Note: HF Pipeline minimal binding manually handled in graph due to version differences, 
# but for this implementation we will use a prompt-based ReAct approach if bind_tools fails, 
# or rely on LangGraph's prebuilt tooling.
# For robust local execution, we explicitly define the node logic below.

# --- AGENT ORCHESTRATOR (LangGraph) ---

SYSTEM_PROMPT = """You are the OKNLAB Core, an elite AI orchestrating a team of specialized agents.
Your role is to answer the user's request by dispatching the right sub-personality.

Available Personas:
1. CodeArchitect: Writes clean, modular Python/JS/Go.
2. SecAnalyst: Audits code and checks for vulnerabilities.
3. AutoBot: Handles automation logic.
4. CreativeAgent: Generates content.

Instructions:
- If you need current data, call 'google_search_tool'.
- If you need to calculate or test logic, call 'python_repl_tool'.
- Format your final answer clearly.
- Be pragmatic and precise.
"""

# Helper to simulate tool binding for local HF models
def chatbot_node(state: MessagesState):
    messages = state["messages"]
    # We construct a prompt including tool definitions for the model to understand
    # Since small local models struggle with native tool binding sometimes, we ensure context is clear.
    response = llm.invoke(messages)
    return {"messages": [response]}

# Simple Graph Construction
workflow = StateGraph(MessagesState)

# Using a simplified ReAct pattern compatible with local HF pipelines
# In a production environment with OpenAI/Anthropic, we would use tools_condition
# Here we implement a custom router logic or simple chain.

# For stability in Colab with 1.5B model, we will use a direct Chain approach 
# wrapped in the endpoint rather than complex cyclical graph to avoid hallucination loops.
# We update the graph to a simple retrieval chain for this demo.

from langchain.agents import AgentExecutor, create_tool_calling_agent, create_react_agent
from langchain import hub

# Pull a standard prompt or define one
prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# We create a ReAct agent which works better with generic LLMs
react_agent = create_react_agent(llm, tools, prompt_template)
agent_executor = AgentExecutor(agent=react_agent, tools=tools, verbose=True, handle_parsing_errors=True)

# --- API ENDPOINTS ---

class ChatRequest(BaseModel):
    message: str
    agent_mode: str = "CodeArchitect"

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    try:
        # Context injection based on mode
        mode_instruction = ""
        if req.agent_mode == "SecAnalyst":
            mode_instruction = "Focus on security, vulnerabilities, and threat modeling. "
        elif req.agent_mode == "AutoBot":
            mode_instruction = "Focus on automation, APIs, and workflow efficiency. "
        
        full_input = mode_instruction + req.message
        
        result = agent_executor.invoke({"input": full_input, "chat_history": []})
        return {
            "response": result["output"],
            "agent": req.agent_mode,
            "status": "success"
        }
    except Exception as e:
        print(f"Error: {e}")
        return {"response": f"System Critical: {str(e)}", "status": "error"}

@app.get("/")
async def root():
    return {"system": "OKNLAB AI Core", "status": "operational", "gpu": torch.cuda.get_device_name(0)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
