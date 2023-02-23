# load gpt3 in openai and generate a completion to question "Write a title for a movie" and print it out
import openai
import os

# Open json entitled /shared/ademi_adeniji/something-something/something-something-v2-labels.json
with open('/shared/ademi_adeniji/something-something/something-something-v2-labels.json') as json_file:
    # iterate through it row by row skipping the first row and the last row
    for row in json_file.readlines()[1:-1]:
        # extract string between first pair of double quotes
        label = row.split('"')[1]
    
        prompt = f"""Generate 10 simple, action-oriented instructional sentences
        that pertain to manipulation of objects and are
        semantically opposite from "{label}" 
        where the only noun in the sentence is still "something" and don't include 
        uncommon objects. Vary the verb tense used in each sentence. All lowercase.
        Comma separate the sentences on a single line."""

        openai.api_key = os.getenv("OPENAI_API_KEY")
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0.1,
            max_tokens=700,
            top_p=1,
        )
        print(label)
        print(response["choices"][0]["text"])

        prompt = f"""Generate 10 simple, action-oriented instructional sentences
        that pertain to manipulation of objects and are
        semantically the same as "{label}" 
        where the only noun in the sentence is still "something" and don't include 
        uncommon objects. Make sure "something" is still in the sentence.
        Vary the verb tense used in each sentence. All lowercase.
        Comma separate the sentences on a single line."""

        openai.api_key = os.getenv("OPENAI_API_KEY")
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0.1,
            max_tokens=700,
            top_p=1,
        )
        print(response["choices"][0]["text"])
        # print a space
        print()




   







    
    
        



        