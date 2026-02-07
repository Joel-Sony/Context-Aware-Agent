import os
import time
from pinecone import Pinecone
from dotenv import load_dotenv

# 1. Load Environment Variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")

# 2. Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=INDEX_HOST)

# 3. Define Guidelines
guidelines_data = [
    # --- ORIGINAL 10 (REWRITTEN FOR BETTER SIMILARITY) ---
    {"id": "mh_panic_001", "category": "Panic Attack", "text": "I can't breathe, my heart is racing, I feel like I'm dying, panic attack, shortness of breath, chest tightness, something terrible is happening.", "measures": "5-4-3-2-1 Grounding: 5 see, 4 touch, 3 hear, 2 smell, 1 taste.", "risk": "High"},
    {"id": "mh_anxiety_002", "category": "General Anxiety", "text": "I am so worried, I can't stop thinking about what might go wrong, chronic anxiety, overthinking, nervous tension, restless mind.", "measures": "Box Breathing: Inhale 4s, hold 4s, exhale 4s, hold 4s.", "risk": "Medium"},
    {"id": "mh_depression_003", "category": "Depressive Episode", "text": "I feel so low, I have no energy, everything feels pointless, depressed, hopeless, sadness, worthlessness, can't get out of bed.", "measures": "Behavioral Activation: Set one tiny task like drinking water.", "risk": "Medium"},
    {"id": "mh_burnout_004", "category": "Burnout", "text": "I am exhausted by work, I have no motivation left, burnout, work stress, emotional drain, professional fatigue, drained.", "measures": "Digital Detox: Disconnect for 30 minutes.", "risk": "Low"},
    {"id": "mh_social_005", "category": "Social Anxiety", "text": "I'm afraid of people judging me, I hate being watched, social phobia, fear of public speaking, embarrassment, social awkwardness.", "measures": "External Focusing: Describe objects in the room.", "risk": "Low"},
    {"id": "mh_grief_006", "category": "Grief & Loss", "text": "I lost someone, I miss them so much, grief, mourning, bereavement, deep sorrow, broken heart, loss of a loved one.", "measures": "Self-Compassion: Allow yourself to feel without judgment.", "risk": "Medium"},
    {"id": "mh_ptsd_007", "category": "Trauma Trigger", "text": "I keep having flashbacks, I feel unsafe, trauma trigger, PTSD, distressing memories, hypervigilance, past trauma.", "measures": "Heel-to-Toe Walking: Focus strictly on physical sensation.", "risk": "High"},
    {"id": "mh_insomnia_008", "category": "Sleep Distress", "text": "I can't sleep, my mind is racing at night, insomnia, sleep deprivation, awake at 3am, can't fall asleep.", "measures": "Cognitive Shuffling: Visualize words starting with A, then B.", "risk": "Low"},
    {"id": "mh_self_esteem_009", "category": "Low Self-Esteem", "text": "I hate myself, I'm not good enough, low self-confidence, self-criticism, insecurity, feeling like a failure.", "measures": "Affirmation: Acknowledge one small success today.", "risk": "Low"},
    {"id": "mh_crisis_010", "category": "Acute Crisis", "text": "I can't cope anymore, I am in crisis, total breakdown, overwhelmed to the point of danger, mental health emergency.", "measures": "Immediate Referral: Contact a crisis hotline or ER.", "risk": "High"},

    # --- NEW 10 (REWRITTEN FOR BETTER SIMILARITY) ---
    {"id": "mh_ocd_011", "category": "Obsessive Thoughts", "text": "I have these intrusive thoughts, I keep repeating things, OCD, compulsions, repetitive urges, intrusive mind loops.", "measures": "Exposure Prevention: Label the thought as 'just a thought' and delay the compulsion.", "risk": "Medium"},
    {"id": "mh_anger_012", "category": "Anger Management", "text": "I am so angry, I want to scream, rage, frustration, losing my temper, out of control anger, feeling explosive.", "measures": "Temperature Shock: Splash cold water on your face or hold an ice cube.", "risk": "Low-Medium"},
    {"id": "mh_phobia_013", "category": "Specific Phobia", "text": "I am terrified of this specific thing, irrational fear, phobia, extreme fright, triggered by a situation or object.", "measures": "Progressive Muscle Relaxation: Tense and release muscle groups.", "risk": "Low"},
    {"id": "mh_adhd_014", "category": "Executive Dysfunction", "text": "I feel paralyzed, I can't start my work, task paralysis, procrastinating, ADHD, distracted, can't focus.", "measures": "The 5-Second Rule: Count down 5-4-3-2-1 and move on '1'.", "risk": "Low"},
    {"id": "mh_bipolar_015", "category": "Mood Instability", "text": "My mood is swinging up and down, extreme highs and lows, bipolar, mood swings, energy shifts, instability.", "measures": "Routine Anchor: Focus on a fixed sleep schedule and mood tracking.", "risk": "Medium-High"},
    {"id": "mh_eating_016", "category": "Body Dysmorphia", "text": "I hate how I look in the mirror, eating disorder thoughts, body image issues, dysmorphia, weight obsession.", "measures": "Body Neutrality: List three functional things your body did for you today.", "risk": "High"},
    {"id": "mh_lonely_017", "category": "Chronic Loneliness", "text": "I am so alone, nobody cares about me, isolation, loneliness, feeling disconnected, social withdrawal.", "measures": "Low-Stakes Interaction: Engage in a 'micro-interaction' like saying hello.", "risk": "Medium"},
    {"id": "mh_work_018", "category": "Workplace Stress", "text": "Work is killing me, deadlines are too much, toxic boss, career pressure, professional anxiety.", "measures": "Time Boxing: Focus for 25 minutes, then disconnect for 5.", "risk": "Low"},
    {"id": "mh_health_019", "category": "Health Anxiety", "text": "I think I have a serious disease, health anxiety, hypochondria, googling symptoms, afraid of being sick.", "measures": "Fact-Checking: List physical sensation vs. catastrophic thought.", "risk": "Low-Medium"},
    {"id": "mh_guilt_020", "category": "Chronic Guilt", "text": "Everything is my fault, I feel so guilty, regret, blaming myself for things I can't control.", "measures": "Responsibility Pie Chart: Assign percentages to all factors involved.", "risk": "Low"}
]

print(f"Starting single-record upsert for {len(guidelines_data)} guidelines...")

# 4. Loop and Upsert One-by-One
for item in guidelines_data:
    try:
        # NOTE: Using upsert_records for Serverless Inference
        # Pinecone uses the '_id' field and the 'text' field (or whatever your field_map is)
        index.upsert_records(
            namespace="medical-guidelines",
            records=[{
                "_id": item["id"],
                "text": item["text"], # This must match your index field_map
                "category": item["category"],
                "measures": item["measures"],
                "risk_level": item["risk"]
            }]
        )
        print(f"✅ Successfully uploaded: {item['id']}")
        
        # Small sleep to prevent hitting rate limits during individual calls
        time.sleep(0.5) 
        
    except Exception as e:
        print(f"❌ Error uploading {item['id']}: {e}")

print("\nUpload process finished.")