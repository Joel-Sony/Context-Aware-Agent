from pinecone import Pinecone
import os 
from dotenv import load_dotenv
import numpy as np

load_dotenv()


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "temp"

index = pc.Index(index_name)

# # List of sentences to upsert
# sentences = [
# "What are the main points I emphasized?",
#     "In a nutshell, what did we cover?",
#     "Recap the progress we've made so far.",
#     "Give me the short version of our dialogue.",
#     "What's the gist of my feedback?",
#     "Summarize the complex idea I explained.",
#     "Provide a quick rundown of the requirements.",
#     "What were the most important facts shared?",
#     "Give me the condensed version of the notes.",
#     "What is the overall theme of our conversations?",
#     "Summarize the history of my complaints.",
#     "What were the various options we considered?",
#     "Synthesize the different opinions I gave you.",
#     "What does the data we shared collectively suggest?",
#     "Provide a general overview of the past week's activity.",
#     "What is the simplified version of the steps?",
#     "Summarize my stated budget constraints.",
#     "What were the pros and cons we listed?",
#     "Give me the three main pillars of my argument.",
#     "What are the key decisions I have made?",
#     "Summarize the details of my recent trip.",
#     "What was the conclusion of our previous analysis?",
#     "Give me a bullet-point list of the things I said.",
#     "What were the final consensus points?",
#     "If I said that, what was the next step we planned?",
#     "Based on my last answer, what should I do now?",
#     "If I recall correctly, we set a deadline. What was it?",
#     "Given the information I provided, what is the recommendation?",
#     "Assuming what I told you is still true, what changes?",
#     "Using my previously stated goals, create a new plan.",
#     "Considering my budget, what did we decide to purchase?",
#     "Since I already told you my address, can you use it?",
#     "What was the contingency plan we made?",
#     "In the event that 'X' happened (as discussed), what is the response?",
#     "If my situation is what I described earlier, what's the outcome?",
#     "Where does this stand in relation to the timeline we created?",
#     "How do I proceed from the point we left off?",
#     "Given my history, is this a good decision?",
#     "What were the next three actions I committed to?",
#     "What was the outcome of the query I ran last time?",
#     "If my preference is 'Y', then what is the implication?",
#     "What steps did we plan for the next phase?",
#     "If I follow the original strategy, what happens?",
#     "What was the original context for this follow-up?",
#     "If I told you 'A', what was your corresponding response 'B'?",
#     "What did you predict based on the facts I gave you?",
#     "Using the criteria we established, which option is best?",
#     "How can I incorporate my previous feedback?",
#     "What was the purpose of the file I uploaded?",
#     "If I were to change one variable, what were the existing ones?",
#     "What were the parameters we defined for success?",
#     "Assuming my schedule hasn't changed, when is the meeting?",
#     "What was the reason we decided against that earlier?",
#     "Given the initial premises, what is the logical next move?",
#     "I need the details of my insurance policy I shared.",
#     "Find the code snippet I posted about Python recursion.",
#     "What was the exact quote I used from the movie?",
#     "Retrieve the specific statistics I mentioned about the climate.",
#     "Look up the financial figures I input.",
#     "What were the three items on my grocery list?",
#     "Find the source or link I sent you.",
#     "What was the definition of that complex term?",
#     "Can you pull up the notes on my fantasy league team?",
#     "I need the specific model number of the appliance.",
#     "Find the list of reasons why I hate olives.",
#     "What was the password hint I mentioned?",
#     "Retrieve the detailed itinerary we planned.",
#     "What are the measurements for the DIY project?",
#     "Find the complex formula I typed out.",
#     "What were the names of the historical figures discussed?",
#     "Look up the exact phrasing of my thesis statement.",
#     "What were the ingredients for the dish I was cooking?",
#     "I need the reference number from that document.",
#     "Find the error message I showed you last time.",
#     "What was the exact title of the article I referenced?",
#     "Retrieve the different color options we discussed.",
#     "What were the delivery instructions I gave?",
#     "Can you find the budget breakdown?",
#     "What was the legal term I asked about?",
#     "Find the list of suggested improvements.",
#     "What are the current settings for my profile?",
#     "I need the name of the doctor I visited.",
#     "What was the serial number for that device?",
#     "Retrieve the image description I provided."
# ]

# # Prepare the sentences for upsertion
# upsert_data = [{"id": str(i), "text": sentence} for i, sentence in enumerate(sentences, start = 97)]

# # Upsert the sentences into Pinecone
# index.upsert_records("namespace",upsert_data)

triggerEmbeddings = []

# Retrieve the embeddings for each sentence
ids_to_fetch = [str(i) for i in range(1,180)]
 
res = index.fetch(ids= ids_to_fetch, namespace="namespace")

for i in range(1,180):
    id = str(i)
    triggerEmbeddings.append(res.vectors[id].values)

# print(triggerEmbeddings)

# Convert list of embeddings to numpy array
embeddings_array = np.array(triggerEmbeddings)

# Save to file
np.save("embeddings.npy", embeddings_array)

# Load back later
loaded_embeddings = np.load("embeddings.npy")
print(loaded_embeddings.shape) 