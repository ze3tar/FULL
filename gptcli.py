#!/usr/bin/env python3
import os
import sys
from openai import OpenAI

def read_project_files(base_dir="."):
    file_contents = {}
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith(('.py', '.txt', '.md', '.c', '.cpp', '.js', '.html', '.css')):
                path = os.path.join(root, f)
                try:
                    with open(path, "r", encoding="utf-8") as file:
                        file_contents[path] = file.read()
                except Exception as e:
                    print(f"Error reading {path}: {e}")
    return file_contents

def chat_with_gpt(prompt, context):
    # Create a client using the NEW API
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ No API key found. Please export OPENAI_API_KEY first.")
        sys.exit(1)

    response = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": f"Project context:\n{context}\n\nUser request:\n{prompt}"}
        ]
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: ./gptcli.py 'Your instruction here'")
        sys.exit(1)

    prompt = sys.argv[1]
    project_context = "\n".join(
        f"{name}:\n{content[:1000]}"
        for name, content in read_project_files().items()
    )

    reply = chat_with_gpt(prompt, project_context)
    print("\n=== GPT RESPONSE ===\n")
    print(reply)
