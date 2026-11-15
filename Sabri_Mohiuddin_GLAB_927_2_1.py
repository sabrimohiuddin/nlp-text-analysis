import nltk

# Download the English data and models needed for this lab.
for resource in (
    "punkt_tab",
    "averaged_perceptron_tagger_eng",
    "maxent_ne_chunker_tab",
    "words",
):
    nltk.download(resource, quiet=True, raise_on_error=True)

business_document = """
The recent feedback from Acme Corp highlighted issues with the new model of the Omega widget.
Customers from New York and San Francisco have reported delays in shipping.
The CEO of Acme Corp, Jane Doe, mentioned plans to address these concerns by Q3.
"""

# Split the feedback into tokens and label each token's grammatical role.
tokens = nltk.word_tokenize(business_document)
pos_tags = nltk.pos_tag(tokens)

# Task 1: Find and print named entities, such as companies, people, and places.
# This can help a business see who and what customers mention in their feedback.
ner_result = nltk.ne_chunk(pos_tags)
entities = []

print("Named Entities:")
for entity in ner_result:
    if isinstance(entity, nltk.Tree):
        entity_name = " ".join(word for word, tag in entity.leaves())
        entities.append((entity_name, entity.label()))
        print(entity_name)

# Task 2: Print the category NLTK predicted for each detected name.
print("\nNamed Entities with Categories:")
for entity_name, entity_category in entities:
    print(f"{entity_name}: {entity_category}")

# Task 3: Print every token with its part-of-speech tag.
print("\nWords and POS Tags:")
for word, tag in pos_tags:
    print(f"{word}: {tag}")

    