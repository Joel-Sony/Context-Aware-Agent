import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=INDEX_HOST)

# Your Mental Health Guidelines
guidelines = [
    {"_id": "mh_work_018", "category": "Workplace Stress", "measures": "Time Boxing: Focus for 25 minutes, then disconnect for 5.", "risk_level": "Low", "text": "Work is killing me, deadlines are too much, toxic boss, career pressure, professional anxiety."},
    {"_id": "mh_burnout_004", "category": "Burnout", "measures": "Digital Detox: Disconnect for 30 minutes.", "risk_level": "Low", "text": "I am exhausted by work, I have no motivation left, burnout, work stress, emotional drain, professional fatigue, drained."},
    {"_id": "mh_depression_003", "category": "Depressive Episode", "measures": "Behavioral Activation: Set one tiny task like drinking water.", "risk_level": "Medium", "text": "I feel so low, I have no energy, everything feels pointless, depressed, hopeless, sadness, worthlessness, can't get out of bed."},
    {"_id": "mh_grief_006", "category": "Grief & Loss", "measures": "Self-Compassion: Allow yourself to feel without judgment.", "risk_level": "Medium", "text": "I lost someone, I miss them so much, grief, mourning, bereavement, deep sorrow, broken heart, loss of a loved one."},
    {"_id": "mh_self_esteem_009", "category": "Low Self-Esteem", "measures": "Affirmation: Acknowledge one small success today.", "risk_level": "Low", "text": "I hate myself, I'm not good enough, low self-confidence, self-criticism, insecurity, feeling like a failure."},
    {"_id": "mh_phobia_013", "category": "Specific Phobia", "measures": "Progressive Muscle Relaxation: Tense and release muscle groups.", "risk_level": "Low", "text": "I am terrified of this specific thing, irrational fear, phobia, extreme fright, triggered by a situation or object."},
    {"_id": "mh_anxiety_002", "category": "General Anxiety", "measures": "Box Breathing: Inhale 4s, hold 4s, exhale 4s, hold 4s.", "risk_level": "Medium", "text": "I am so worried, I can't stop thinking about what might go wrong, chronic anxiety, overthinking, nervous tension, restless mind."},
    {"_id": "mh_panic_001", "category": "Panic Attack", "measures": "5-4-3-2-1 Grounding: 5 see, 4 touch, 3 hear, 2 smell, 1 taste.", "risk_level": "High", "text": "I can't breathe, my heart is racing, I feel like I'm dying, panic attack, shortness of breath, chest tightness, something terrible is happening."},
    {"_id": "mh_eating_016", "category": "Body Dysmorphia", "measures": "Body Neutrality: List three functional things your body did for you today.", "risk_level": "High", "text": "I hate how I look in the mirror, eating disorder thoughts, body image issues, dysmorphia, weight obsession."},
    {"_id": "mh_bipolar_015", "category": "Mood Instability", "measures": "Routine Anchor: Focus on a fixed sleep schedule and mood tracking.", "risk_level": "Medium-High", "text": "My mood is swinging up and down, extreme highs and lows, bipolar, mood swings, energy shifts, instability."},
    {"_id": "mh_adhd_014", "category": "Executive Dysfunction", "measures": "The 5-Second Rule: Count down 5-4-3-2-1 and move on '1'.", "risk_level": "Low", "text": "I feel paralyzed, I can't start my work, task paralysis, procrastinating, ADHD, distracted, can't focus."},
    {"_id": "mh_ocd_011", "category": "Obsessive Thoughts", "measures": "Exposure Prevention: Label the thought as 'just a thought' and delay the compulsion.", "risk_level": "Medium", "text": "I have these intrusive thoughts, I keep repeating things, OCD, compulsions, repetitive urges, intrusive mind loops."},
    {"_id": "mh_lonely_017", "category": "Chronic Loneliness", "measures": "Low-Stakes Interaction: Engage in a 'micro-interaction' like saying hello.", "risk_level": "Medium", "text": "I am so alone, nobody cares about me, isolation, loneliness, feeling disconnected, social withdrawal."},
    {"_id": "mh_guilt_020", "category": "Chronic Guilt", "measures": "Responsibility Pie Chart: Assign percentages to all factors involved.", "risk_level": "Low", "text": "Everything is my fault, I feel so guilty, regret, blaming myself for things I can't control."},
    {"_id": "mh_insomnia_008", "category": "Sleep Distress", "measures": "Cognitive Shuffling: Visualize words starting with A, then B.", "risk_level": "Low", "text": "I can't sleep, my mind is racing at night, insomnia, sleep deprivation, awake at 3am, can't fall asleep."},
    {"_id": "mh_anger_012", "category": "Anger Management", "measures": "Temperature Shock: Splash cold water on your face or hold an ice cube.", "risk_level": "Low-Medium", "text": "I am so angry, I want to scream, rage, frustration, losing my temper, out of control anger, feeling explosive."},
    {"_id": "mh_social_005", "category": "Social Anxiety", "measures": "External Focusing: Describe objects in the room.", "risk_level": "Low", "text": "I'm afraid of people judging me, I hate being watched, social phobia, fear of public speaking, embarrassment, social awkwardness."},
    {"_id": "mh_health_019", "category": "Health Anxiety", "measures": "Fact-Checking: List physical sensation vs. catastrophic thought.", "risk_level": "Low-Medium", "text": "I think I have a serious disease, health anxiety, hypochondria, googling symptoms, afraid of being sick."},
    {"_id": "mh_ptsd_007", "category": "Trauma Trigger", "measures": "Heel-to-Toe Walking: Focus strictly on physical sensation.", "risk_level": "High", "text": "I keep having flashbacks, I feel unsafe, trauma trigger, PTSD, distressing memories, hypervigilance, past trauma."},
    {"_id": "mh_crisis_010", "category": "Acute Crisis", "measures": "Immediate Referral: Contact a crisis hotline or ER.", "risk_level": "High", "text": "I can't cope anymore, I am in crisis, total breakdown, overwhelmed to the point of danger, mental health emergency."}
]

# Upsert records into the 'medical-guidelines' namespace
index.upsert_records("medical-guidelines", guidelines)
print("Successfully uploaded mental health guidelines.")