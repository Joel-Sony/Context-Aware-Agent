from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(host=os.getenv("PINECONE_INDEX_HOST"))

index.upsert_records(
    namespace="medical_guidelines",
    records=[
        {
            "_id": "chest_pain_emergency",
            "text": "Chest pain combined with symptoms such as dizziness, shortness of breath, sweating, nausea, or pain spreading to the arm, back, neck, or jaw may indicate a serious medical emergency and requires immediate medical attention.",
            "risk": "high",
            "category": "cardiac"
        },
        {
            "_id": "breathing_difficulty_emergency",
            "text": "Difficulty breathing, persistent shortness of breath, wheezing, or a sensation of not getting enough air can be life-threatening and should be evaluated urgently by medical professionals.",
            "risk": "high",
            "category": "respiratory"
        },
        {
            "_id": "severe_headache_emergency",
            "text": "A sudden and severe headache, especially one described as the worst headache ever or accompanied by confusion, vision problems, weakness, or difficulty speaking, may require immediate medical evaluation.",
            "risk": "high",
            "category": "neurological"
        },
        {
            "_id": "loss_of_consciousness",
            "text": "Fainting, sudden loss of consciousness, seizures, or unresponsiveness can be signs of a serious underlying condition and warrant urgent medical assessment.",
            "risk": "high",
            "category": "neurological"
        },
        {
            "_id": "stroke_warning_signs",
            "text": "Sudden weakness on one side of the body, facial drooping, slurred speech, confusion, or difficulty understanding speech may indicate a stroke and requires immediate emergency care.",
            "risk": "high",
            "category": "stroke"
        },
        {
            "_id": "self_harm_risk",
            "text": "Thoughts of self-harm, feeling unsafe, or expressing a desire to harm oneself require immediate support. Contacting emergency services or reaching out to a trusted person is strongly advised.",
            "risk": "high",
            "category": "mental_health"
        },
        {
            "_id": "persistent_fever",
            "text": "A fever lasting more than three days, or a fever accompanied by rash, confusion, severe pain, dehydration, or difficulty breathing, should be evaluated by a healthcare professional.",
            "risk": "medium",
            "category": "infection"
        },
        {
            "_id": "abdominal_pain",
            "text": "Severe, worsening, or persistent abdominal pain, especially when associated with vomiting, fever, swelling, or blood in stool, may require medical evaluation.",
            "risk": "medium",
            "category": "gastrointestinal"
        },
        {
            "_id": "vomiting_blood",
            "text": "Vomiting blood or passing black, tarry stools can indicate internal bleeding and should be evaluated urgently by medical professionals.",
            "risk": "high",
            "category": "bleeding"
        },
        {
            "_id": "uncontrolled_bleeding",
            "text": "Uncontrolled bleeding, deep wounds, or significant injuries that do not stop bleeding with pressure should be assessed by a medical professional immediately.",
            "risk": "high",
            "category": "trauma"
        },
        {
            "_id": "head_injury",
            "text": "Head injuries followed by confusion, vomiting, loss of consciousness, severe headache, or changes in behavior should be evaluated by a healthcare professional.",
            "risk": "high",
            "category": "head_injury"
        },
        {
            "_id": "mental_health_distress",
            "text": "Persistent feelings of sadness, anxiety, panic, or emotional distress that interfere with daily functioning may benefit from support from a mental health professional.",
            "risk": "medium",
            "category": "mental_health"
        },
        {
            "_id": "panic_attack",
            "text": "Sudden episodes of intense fear accompanied by chest discomfort, rapid heartbeat, shortness of breath, or dizziness may resemble panic attacks, but medical evaluation is recommended if symptoms are new or severe.",
            "risk": "medium",
            "category": "anxiety"
        },
        {
            "_id": "allergic_reaction",
            "text": "Swelling of the face, lips, tongue, or throat, difficulty breathing, hives, or dizziness following exposure to a substance may indicate a severe allergic reaction and requires immediate medical attention.",
            "risk": "high",
            "category": "allergy"
        },
        {
            "_id": "mild_symptoms",
            "text": "Mild symptoms such as common cold, temporary fatigue, or minor aches often resolve on their own, but medical advice may be helpful if symptoms worsen or persist.",
            "risk": "low",
            "category": "general"
        }
    ]
)

print("✅ Medical guidelines inserted using Pinecone integrated embeddings.")


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

# triggerEmbeddings = []

# # Retrieve the embeddings for each sentence
# ids_to_fetch = [str(i) for i in range(1,180)]
 
# res = index.fetch(ids= ids_to_fetch, namespace="namespace")

# for i in range(1,180):
#     id = str(i)
#     triggerEmbeddings.append(res.vectors[id].values)

# # print(triggerEmbeddings)

# # Convert list of embeddings to numpy array
# embeddings_array = np.array(triggerEmbeddings)

# # Save to file
# np.save("embeddings.npy", embeddings_array)

# # Load back later
# loaded_embeddings = np.load("embeddings.npy")
# print(loaded_embeddings.shape) 