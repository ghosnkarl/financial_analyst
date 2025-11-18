from typing import List

from crewai import LLM, Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task

from .tools.schema_mapper_tool import SchemaMapperTool
from .tools.batch_processor_tool import BatchProcessorTool
from .tools.csv_analyzer_tool import CsvAnalyzerTool
from .tools.visualization_tool import VisualizationTool


llm = LLM(
    model="phi4:14b",
    base_url="http://192.168.2.82:11434/v1",
)

# coder_llm = LLM(
#     model="qwen3-coder:30b-a3b-instruct-q4_k_m",
#     base_url="http://192.168.2.82:11434/v1",
# )


@CrewBase
class FinancialAnalyst:
    """FinancialAnalyst crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def transaction_processor_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["transaction_processor_agent"],
            verbose=True,
            llm=llm,
            tools=[SchemaMapperTool(), BatchProcessorTool()],
        )

    @agent
    def visualization_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["visualization_agent"],
            verbose=True,
            llm=llm,
            tools=[VisualizationTool()],
        )

    @agent
    def data_analyst_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["data_analyst_agent"],
            verbose=True,
            llm=llm,
            tools=[CsvAnalyzerTool()],
        )

    @task
    def process_and_standardize_transactions(self) -> Task:
        return Task(
            config=self.tasks_config["process_and_standardize_transactions"],
        )

    @task
    def generate_visualization_charts(self) -> Task:
        return Task(
            config=self.tasks_config["generate_visualization_charts"],
        )

    @task
    def analyze_transactions(self) -> Task:
        return Task(
            config=self.tasks_config["analyze_transactions"],
            output_file="output/analyze_transactions.md",
        )

    @crew
    def crew(self) -> Crew:
        """Creates the FinancialAnalyst crew"""

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
