from typing import List

from acp_sdk.models import Message, Metadata
from acp_sdk.server import Context, RunYield, RunYieldResume, Server

from crewai import LLM

server = Server()

@server.agent()
def oneliner(input: List[Message]):
    llm = LLM(
        model=f"openai/{os.getenv('LLM_MODEL', 'llama3.1')}",
        base_url=os.getenv("LLM_API_BASE", "http://localhost:11434/v1"),
        api_key=os.getenv("LLM_API_KEY", "dummy"),
    )

    try:



