from openai import OpenAI

client = OpenAI(api_key="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")  # use your real key

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Say hello"}]
)

print(response.choices[0].message.content)