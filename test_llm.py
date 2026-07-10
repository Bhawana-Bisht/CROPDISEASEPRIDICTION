import time
from src.predict.llm_module import generate_llm_response

start = time.time()

response = generate_llm_response(
    "Bacterialblight",
    "Leaves are turning yellow",
    "English"
)

end = time.time()

print(response)
print(f"\nResponse Time: {end-start:.2f} seconds")