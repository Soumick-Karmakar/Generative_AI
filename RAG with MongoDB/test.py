# from google import genai
# import os
# from dotenv import load_dotenv

# load_dotenv()

# client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# response = client.models.embed_content(
#     model="gemini-embedding-001",
#     contents="Hello World"
# )

# print("Embedding Success")
# print(len(response.embeddings[0].values))



from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Say hello in one sentence"
)

print(response.text)