from crewai import Crew, Agent, Task, LLM, llm
import litserve as ls
from crewai_tools import FirecrawlWebSearchTool, VectorDBSearchTool
from langchain_community.llms import Ollama



llm = Ollama(model='qwen3:8b')
llm1 = Ollama(model='deepseek-r1:8b')


class AgenticRAGAPI(ls.LitAPI):
    def setup(self, device):
        #llm = LLM(model='ollama/qwen3:8b')
        #llm1 = LLM(model='ollama/deepseek-r1:8b') 
    
        research_agent = Agent(
            role = 'Senior Researcher',
            goal = 'Research about the user query',
            backstory = 'You are a senior researcher with a knack for finding information',
            tools = [FirecrawlWebSearchTool(), VectorDBSearchTool()],
            llm = llm
        )
        
        research_task = Task(
            description = 'Research about: {query}',
            expected_output = 'A detailed report about the user query',
            agent = research_agent
        )

        writer_agent = Agent(
            role = 'Senior Writer',
            goal = 'use the detailed report to answer the user query',
            backstory = 'You are a very skilled senior writer with a knack for writing',
            llm = llm1
        )

        writer_task = Task(
            description = 'Use the insights from the research to answer: {query}',
            expected_output = 'A detailed answer to the user query',
            agent = writer_agent
        )

        self.crew = Crew(
            agents = [research_agent, writer_agent],
            tasks = [research_task, writer_task]
        )
    
    def decode_request(self, request):
        return request["Query"]

    def predict(self, query):
        return self.crew.kickoff(inputs = {'Query': query})

    def encode_response(self, output):
        return {"output": output}
        
if __name__ == "__main__":
    # To run this server: litserve run server
    server = ls.LitServer(AgenticRAGAPI(), port=8000)
    server.run()