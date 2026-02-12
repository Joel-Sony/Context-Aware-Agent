from openai import OpenAI

# Initialize the client to point to your local machine
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama" # Required by the library, but ignored by Ollama
)

print("--- Testing Connection to Local Ollama ---")

try:
    response = client.chat.completions.create(
        model="gemma2:2b",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Lenni is online' if you can hear me."}
        ],
        max_tokens=20
    )
    print(f"Response: {response.choices[0].message.content}")
    print("\n✅ Success! Your local LLM is talking back.")

except Exception as e:
    print(f"\n❌ Connection Failed: {e}")
    print("Ensure Ollama is running in your taskbar and you ran 'ollama pull gemma2:2b'")