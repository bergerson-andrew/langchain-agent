from langchain_azure_ai.chat_models import AzureAIOpenAIApiChatModel
from azure.identity import DefaultAzureCredential
from langchain_community.agent_toolkits.load_tools import load_tools
import os

try:
	# Newer LangChain
	from langchain.agents import AgentExecutor, create_react_agent
except ImportError:
	# Older LangChain
	from langchain.agents.agent import AgentExecutor  # type: ignore

	try:
		from langchain.agents.react.agent import create_react_agent  # type: ignore
	except ImportError as e:
		raise ImportError(
			"Could not import `create_react_agent`. Your LangChain version likely doesn't "
			"include the ReAct agent helper. Upgrade LangChain or switch to LangGraph's "
			"prebuilt ReAct agent."
		) from e

try:
	from langchainhub import pull
except ImportError as e:
	raise ImportError(
		"LangChain Hub client not installed. Install it with `pip install langchainhub` "
		"(or add `langchainhub` to requirements.txt)."
	) from e

project_endpoint = os.getenv("AGENT_PROJECT_FOUNDRY_ENDPOINT")
if not project_endpoint:
	raise RuntimeError(
		"Missing env var AGENT_PROJECT_FOUNDRY_ENDPOINT. "
		"Set it to your Azure AI Foundry project endpoint before running."
	)

llm = AzureAIOpenAIApiChatModel(
	project_endpoint=project_endpoint,
	credential=DefaultAzureCredential(),
	model="gpt-5.4-mini"
)

tools = load_tools(["llm-math"], llm=llm)

# 3. Get the "ReAct" prompt template from the Hub
# This tells the LLM to use the "Thought, Action, Observation" loop
prompt = pull("hwchase17/react")

# 4. Create the Agent
agent = create_react_agent(llm, tools, prompt)

# 5. Wrap it in an AgentExecutor to actually run it
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 6. Test it!
response = agent_executor.invoke({"input": "What is the square root of 1234.56 multiplied by 2?"})
print(response["output"])