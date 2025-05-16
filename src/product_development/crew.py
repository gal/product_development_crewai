from typing import List, Iterator
import os
from crewai import LLM, Agent, Crew, Process, Task
from crewai.crew import BaseTool
from crewai.project import CrewBase, agent, crew, task

from acp_sdk import MessagePart, Metadata, Link, LinkType
from acp_sdk.server import Server, Context
from acp_sdk.models import Message

# from langchain_community.tools import DuckDuckGoSearchResults
from crewai_tools import SerperDevTool
from langchain_ibm import ChatWatsonx

# If you want to run a snippet of code before or after the crew starts, 
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators
#
# class SearchTool(BaseTool):
#     name: str = "DuckDuckGo Search Tool"
#     description: str = "Search the web for a given query."
#
#     def _run(self, query: str) -> str:
#         # Ensure the DuckDuckGoSearchRun is invoked properly.
#         duckduckgo_tool = DuckDuckGoSearchResults()
#         response = duckduckgo_tool.invoke(query)
#         return response

#     def _get_tool(self):
#         # Create an instance of the tool when needed
#         return SearchTool()


@CrewBase
class ProductDevelopment():
    """ProductDevelopment crew"""

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    
    @agent 
    def feature_focused_product_manager(self) -> Agent:
        return Agent(
            config=self.agents_config['feature_focused_product_manager'], # type: ignore[index]
            verbose=True, backstory=""
            # tools=[DuckDuckGoSearchResults()],
        )

    @agent
    def brand_name_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config['brand_name_specialist'], # type: ignore[index]
            tools=[SerperDevTool()], backstory=""
        )

    @agent
    def copy_writer(self) -> Agent:
        return Agent(
            config=self.agents_config['copy_writer'], # type: ignore[index]
            backstory=""
            # tools=[DuckDuckGoSearchResults()]
        )

    @agent
    def competitor_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['competitor_analyst'], # type: ignore[index]
            tools=[SerperDevTool()], backstory=""
        )

    @agent
    def software_system_architect(self) -> Agent:
        return Agent(
            config=self.agents_config['software_system_architect'], # type: ignore[index]
            backstory=""
        )

    @agent
    def business_analyst_report_writer(self) -> Agent:
        return Agent(
            config=self.agents_config['business_analyst_report_writer'], # type: ignore[index]
            backstory=""
        )


    @task
    def feature_extraction_task(self) -> Task:
        return Task(
            config=self.tasks_config['feature_extraction_task'], # type: ignore[index]
        )

    @task
    def product_name_task(self) -> Task:
        return Task(
            config=self.tasks_config['product_name_task'], # type: ignore[index]
        )

    @task
    def copy_text_task(self) -> Task:
        return Task(
            config=self.tasks_config['copy_text_task'], # type: ignore[index]
        )

    @task
    def competitor_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config['competitor_analysis_task'], # type: ignore[index]
        )

    @task
    def technical_requirements_task(self) -> Task:
        return Task(
            config=self.tasks_config['technical_requirements_task'], # type: ignore[index]
        )

    @task
    def business_analysis_reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['business_analysis_reporting_task'], # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the ProductDevelopment crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True, 
        )


server = Server()


@server.agent(
    metadata=Metadata(
        ui={"type": "hands-off", "user_greeting": "What is the project you would like to analyze?"},
        env=[
            {
                "name": "WATSONX_URL",
                "required": True,
                "description": "Watsonx.ai region."
            },
            {
                "name": "WATSONX_APIKEY",
                "required": True,
                "description": "API Key"
            },
            {
                "name": "WATSONX_INSTANCE_ID",
                "required": True,
                "description": "Project ID"
            }
        ]
    )
)
def product_development_agent(input: List[Message], context: Context) -> Iterator:
    input = {"idea": input[-1].parts[-1].content}
    try:

        llm = ChatWatsonx(
            model_id="ibm/granite-3-2b-instruct",
            url=os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com"),
            project_id="WATSONX_INSTANCE_ID",
        )

        crew = ProductDevelopment().crew()

        crew.chat_llm = llm
        crew.manager_llm = llm
        crew.function_calling_llm = llm

        result = crew.kickoff(inputs={"idea": input})
        print(result)
        yield MessagePart(content=result.raw)
    except:
        raise

def run():
    server.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", 8000)))

if __name__ == "__main__":
    run()
