# llama_pack

from llama_index.core.llama_pack import download_llama_pack
from llama_index.core import SimpleDirectoryReader

from llama_index.readers.file import PDFReader

# PDF Reader with `SimpleDirectoryReader`
parser = PDFReader()
file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader(
    "./data", file_extractor=file_extractor
).load_data()

# download and install dependencies
OllamaQueryEnginePack = download_llama_pack(
    "OllamaQueryEnginePack", "./ollama_pack"
)


# You can use any llama-hub loader to get documents!
ollama_pack = OllamaQueryEnginePack(model="basic_model", documents=documents)

prompt = """Give me an MCQ quiz on this document with 5 questions. 
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
                        
                        """


response = ollama_pack.run(prompt)

print(str(response))