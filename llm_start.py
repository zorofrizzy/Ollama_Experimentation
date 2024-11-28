from llama_index.llms.ollama import Ollama
from time import time
import asyncio


def basic_setup():
    start_time = time()

    llm = Ollama(model="basic_model", request_timeout=120.0, json_mode=True)

    resp = llm.complete("""Give me an MCQ quiz on places to visit in Goa with 5 questions. 
                        Give the answers as well. Output as a structured JSON object.
                        Output in the following JSON format:
    {
        "quiz": [
            {
                "question": "Question text here",
                "options": ["Option1", "Option2", "Option3", "Option4"],
                "answer": "Correct Option"
            },
            ...
        ]
    }
                        
                        """)

    print(str(resp))

    print("\nTime taken : ", round(time() - start_time, 3))


async def async_setup():
    start_time = time()

    # Initialize the LLM
    llm = Ollama(model="basic_model", request_timeout=120.0, json_mode=True)

    # Define the improved prompt
    prompt = """
    Create an MCQ quiz on places to visit in Goa with 5 questions. 
    Provide the answers as well. Output in the following JSON format:
    {
        "quiz": [
            {
                "question": "Question text here",
                "options": ["Option1", "Option2", "Option3", "Option4"],
                "answer": "Correct Option"
            },
            ...
        ]
    }
    """

    try:
        # Use `await` for the async call
        resp = await llm.acomplete(prompt)
        print("Response:\n", str(resp))
    except Exception as e:
        print(f"An error occurred: {e}")

    print("\nTime taken : ", round(time() - start_time, 3))


basic_setup()
#asyncio.run(async_setup())