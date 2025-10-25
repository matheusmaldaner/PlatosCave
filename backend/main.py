from browser_use import Agent, ChatBrowserUse, ChatOpenAI, ChatGoogle, ChatAnthropic, ChatOllama
from browser_use.llm.messages import BaseMessage, UserMessage

from dotenv import load_dotenv
import asyncio
import argparse
from prompts import build_url_paper_analysis_prompt, build_fact_dag_prompt

# remove langchain after
#from langchain_core.messages import HumanMessage

load_dotenv()

async def main(url):
    llm = ChatBrowserUse() # ChatBrowserUse is their in house model
    # alternatively you could use ChatOpenAI(model='o3'), ChatOllama(model="qwen32.1:8b")
    # this would require OPENAI_API_KEY=... , GOOGLE_API_KEY=... , ANTHROPIC_API_KEY=... , 
    
    browsing_url_prompt = build_url_paper_analysis_prompt(paper_url=url)
    agent = Agent(task=browsing_url_prompt, llm=llm)
    #agent = Agent(task="browse matheus.wiki, tell his current school", llm=llm)

    # TODO: make sure it shows interactive elements during the browsing
    history = await agent.run(max_steps=100)
    
    extracted_text = history.final_result()
    with open('extracted_paper_text.txt', 'w', encoding='utf-8') as f:
      f.write(extracted_text)

    # we got all the info about the paper stored in url (all text), extract payload later
    dag_task_prompt = build_fact_dag_prompt(raw_text=extracted_text)
    
    # create the dag from the raw text of the paper, need to pass Message objects
    user_message = UserMessage(content=dag_task_prompt)
    
    with open('user_message.txt', 'w', encoding='utf-8') as f:
      f.write(user_message.text)

    # need to invoke llm with Message objects
    response = await llm.ainvoke(messages=[user_message])

    with open('response_dag.txt', 'w', encoding='utf-8') as f:
      f.write(response.completion)

    # all the data is being stored in temp md files
    # TODO: save the finalized md file so it is not temp

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an agent task with a selected LLM.")
    #parser.add_argument("--task", type=str, help="The task/question for the agent to solve.")
    parser.add_argument("--url", type=str, help="Enter URL to analyze.")

    args = parser.parse_args()

    asyncio.run(main(args.url))
    # to run, use python main.py --url "https://arxiv.org/abs/2305.10403"