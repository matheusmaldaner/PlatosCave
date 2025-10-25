from browser_use import Agent, ChatBrowserUse, ChatOpenAI, Browser, ChatAnthropic, ChatOllama
from browser_use.llm.messages import BaseMessage, UserMessage

from dotenv import load_dotenv
import asyncio
import argparse
from prompts import build_url_paper_analysis_prompt, build_fact_dag_prompt

# remove langchain after
#from langchain_core.messages import HumanMessage

load_dotenv()

async def main(url):
    # unused, will be implemented for the frontend
    browser = Browser(
       cdp_url="http://localhost:9222", # cdp endpoint from noVNC container wip
       headless=False,
       is_local=False,
       keep_alive=True
    )
    
    llm = ChatBrowserUse() # optimized for browser automation w 3-5x speedup 
    # alternatively you could use ChatOpenAI(model='o3'), ChatOllama(model="qwen32.1:8b")
    # this would require OPENAI_API_KEY=... , GOOGLE_API_KEY=... , ANTHROPIC_API_KEY=... , 
    
    browsing_url_prompt = build_url_paper_analysis_prompt(paper_url=url)
    agent = Agent(
       task=browsing_url_prompt,
       llm=llm,
       #browser=browser, # remote or local browser
       vision_detail_level='high',
       generate_gif=True,
       #save_conversation_path='conversation.json',
       use_vision=True)
    #agent = Agent(task="browse matheus.wiki, tell his current school", llm=llm)

    # TODO: make sure it shows interactive elements during the browsing
    history = await agent.run(max_steps=100)
    
    # Get the actual extracted content from the agent's extract actions
    #extracted_text = history.final_result()
    extracted_chunks = [chunk for chunk in history.extracted_content() if chunk]
    extracted_text = "\n\n".join(extracted_chunks)

    # Join all extracted content into a single string (if multiple extractions were made)
    #extracted_text = "\n\n".join(extracted_content) if extracted_content else history.final_result()

    # save for debugging
    with open('extracted_paper_text.txt', 'w', encoding='utf-8') as f:
      f.write(extracted_text)

    # save additional debug info
    browsed_urls = history.urls()
    model_outputs = history.model_outputs()
    last_action = history.last_action()
    with open('extra_data.txt', 'w', encoding='utf-8') as f:
      f.write("Browsed URLs:\n")
      f.writelines(f"{url}\n" for url in browsed_urls)
      # f.write("\nExtracted Content:\n")
      # f.writelines(f"{line}\n" for line in extracted_content)
      f.write("\nModel Outputs:\n")
      f.writelines(f"{line}\n" for line in model_outputs)
      f.write("\nLast Action:\n")
      f.write(str(last_action))

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
    
    with open('final_dag.json', 'w', encoding='utf-8') as f:
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


# spins up multiple parallel agents for the list for dags
# def parallel_run(dag_list):
#     asyncio.run(async_parallel_run(dag_list))