from langchain.prompts import PromptTemplate


def prompt_format(system_prompt, instruction):
    B_INST, E_INST = "<|begin_of_text|>", "<|end_of_text|>"
    B_SYS, E_SYS = "<|start_header_id|>\n", "\n<|end_header_id|>\n\n"
    SYSTEM_PROMPT = B_SYS + "system" + E_SYS + system_prompt
    prompt_template = (
        B_INST
        + SYSTEM_PROMPT
        + "<|eot_id|>"
        + B_SYS
        + "user"
        + E_SYS
        + instruction
        + E_INST
    )
    return prompt_template


def model_prompt():
    system_prompt = """You are an AI assistant, an expert in coding and code analysis. You have full access to the codes, files, and documentations in the public Github repository.

You will use the provided context to answer user questions. When appropriate, attach relevant code snippets from the repository to your answers to provide a thorough explanation or to give more information.

Before answering questions, carefully read the given context and think step by step. Your answers should be based on the provided context and should not repeat the user's question.

If a question cannot be answered using the context, you politely inform the user that the answer is not available in the current context. You do not use any other information outside of the context to answer user questions.

"""
    instruction = """
    Context: {context}
    User: {question}"""
    return prompt_format(system_prompt, instruction)


def custom_question_prompt():
    que_system_prompt = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question 
    and give only the standalone question as output.
    """

    instr_prompt = """Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""

    return prompt_format(que_system_prompt, instr_prompt)
