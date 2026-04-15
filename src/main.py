from langchain_azure_ai.chat_models import AzureAIOpenAIApiChatModel
from azure.identity import DefaultAzureCredential

llm = AzureAIOpenAIApiChatModel(
	project_endpoint=os.environ["AGENT_PROJECT_FOUNDRY_ENDPOINT"],
	credential=os.environ["AGENT_PROJECT_FOUNDRY_KEY"],
	model="gpt-4o"
)

user_prompt = "Explain how Azure Foundry helps build AI agents."

response = llm.invoke(user_prompt)

print(response)