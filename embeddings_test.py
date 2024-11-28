"""
This code is working end to end. 

"""


from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import SimpleDirectoryReader

from llama_index.readers.file import PDFReader
from llama_index.extractors.entity import EntityExtractor
from llama_index.core.node_parser import SentenceSplitter

from llama_index.core.ingestion import IngestionPipeline

from llama_index.core import VectorStoreIndex
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings

##



entity_extractor = EntityExtractor(
    prediction_threshold=0.5,
    label_entities=False,  # include the entity label in the metadata (can be erroneous)
    device="cpu",  # set to "cuda" if you have a GPU
)

node_parser = SentenceSplitter()

transformations = [node_parser, entity_extractor]

##

# PDF Reader with `SimpleDirectoryReader`
parser = PDFReader()
file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader(
    "./data", file_extractor=file_extractor
).load_data()

##
pipeline = IngestionPipeline(transformations=transformations)

nodes = pipeline.run(documents=documents)
##

print("Printing nodes : ", nodes, "\n\nEnd Print Nodes", type(nodes))

ollama_embedding = OllamaEmbedding(
    model_name="basic_model",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
)

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

query_embedding = ollama_embedding.get_query_embedding(prompt)
#print(query_embedding)

Settings.llm = Ollama(model="basic_model", temperature=0.2, json_mode=True)
Settings.embed_model = ollama_embedding


index = VectorStoreIndex(nodes=nodes)

query_engine = index.as_query_engine()
response = query_engine.query(prompt)
print(response)