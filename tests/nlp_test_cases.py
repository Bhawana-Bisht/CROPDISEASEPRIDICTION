from src.nlp_module import extract_symptoms

test_samples = [
    "rice leaf has brown spots and yellow edges",
    "wheat plant showing white powder on leaves",
    "tomato leaves turning black and dry",
    "yellow patches on paddy leaves",
    "leaf curling and holes in plant"
]

for text in test_samples:
    print("\nINPUT:", text)
    print("KEYWORDS:", extract_symptoms(text))