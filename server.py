# server.py
from crewai import Crew, Agent, Task
from crewai_tools import SerperDevTool, WebsiteSearchTool # Using standard tools for this example
import litserve as ls
from langchain_community.llms import Ollama

# It's better practice to initialize the LLM once and pass it to the agents.
# Ensure you have ollama running with both models pulled:
# ollama pull qwen2:7b
# ollama pull deepseek-coder
llm = Ollama(model='qwen2:7b')
llm1 = Ollama(model='deepseek-coder')

# Initialize tools
# Note: SerperDevTool requires a SERPER_API_KEY environment variable.
# You can replace these with your custom Firecrawl/VectorDB tools.
search_tool = SerperDevTool()
web_tool = WebsiteSearchTool()


class AgenticRAGAPI(ls.LitAPI):
    def setup(self, device):
        # Define Agents
        research_agent = Agent(
            role='Senior Researcher',
            goal='Uncover groundbreaking technologies and relevant facts about the user query',
            backstory='You are a master of web research, able to dig up the most relevant and up-to-date information on any topic.',
            tools=[search_tool, web_tool], # Corrected tools
            llm=llm,
            verbose=True
        )

        writer_agent = Agent(
            role='Expert Content Writer',
            goal='Craft a clear, concise, and compelling answer based on the research provided',
            backstory='You are a renowned writer, known for your ability to synthesize complex information into easy-to-understand narratives.',
            llm=llm1,
            verbose=True
        )

        # Define Tasks with dynamic placeholders {query}
        # The placeholder will be filled by the 'inputs' dictionary in crew.kickoff()
        research_task = Task(
            description='Research the user query: "{query}". Find the most critical facts and recent developments.',
            expected_output='A detailed, bullet-point report summarizing the key findings.',
            agent=research_agent
        )

        # IMPORTANT: Add 'context' to pass the output of research_task to this task
        writer_task = Task(
            description='Using the provided research report, write a comprehensive answer to the query: "{query}".',
            expected_output='A well-written, final answer that directly addresses the user\'s query, based on the research.',
            agent=writer_agent,
            context=[research_task] # This creates the dependency and passes data
        )

        # Assemble the crew
        self.crew = Crew(
            agents=[research_agent, writer_agent],
            tasks=[research_task, writer_task],
            verbose=2
        )

    def decode_request(self, request: dict) -> str:
        # Corrected key from "Query" to "query" to match the client
        return request["query"]

    def predict(self, query: str) -> str:
        # The key in the 'inputs' dict must match the placeholder in the tasks
        return self.crew.kickoff(inputs={'query': query})

    def encode_response(self, output: str) -> dict:
        return {"output": output}

if __name__ == "__main__":
    # To run this server: litserve run server
    server = ls.LitServer(AgenticRAGAPI(), port=8000)
    server.run()