

import sys
import os
sys.path.insert(0, 'src')

from main import MahabharatChatbot

print("\n" + "="*80)
print("TESTING IMPROVED RAG PIPELINE")
print("="*80)
print("Features: k=8 retrieval, chunking (3 verses), debug mode, strict grounding")
print("="*80 + "\n")


print("Initializing chatbot with improved configuration...")
chatbot = MahabharatChatbot()

print("\n" + "="*80)
print("TEST 1: Multi-Chapter Comparison Question")
print("="*80)

query1 = "Compare the characteristics of Sthitaprajna from Chapter 2 with the qualities of a Karmayogi in Chapter 5"
print(f"\nQuery: {query1}\n")

result1 = chatbot.query(query1)

if "answer" in result1 and result1["answer"] == "Answer not found":
    print("\n[RESULT] Answer not found")
else:
    print(f"\n[RESULT] Verse Retrieved: {result1.get('verse_number', 'Unknown')}")
    print(f"Confidence: {result1.get('confidence_score', 0)*100:.1f}%")
    
    if "chapters_retrieved" in result1:
        print(f"Chapters Retrieved: {result1['chapters_retrieved']}")
    
    if "llm_response" in result1:
        print("\n--- LLM RESPONSE ---")
        print(result1['llm_response'][:500] + "..." if len(result1['llm_response']) > 500 else result1['llm_response'])
    else:
        print("\n[NO LLM RESPONSE]")

print("\n" + "="*80)
print("TEST 2: Simple Single-Chapter Question")
print("="*80)

query2 = "What is karma yoga?"
print(f"\nQuery: {query2}\n")

result2 = chatbot.query(query2)

if "answer" in result2 and result2["answer"] == "Answer not found":
    print("\n[RESULT] Answer not found")
else:
    print(f"\n[RESULT] Verse: {result2.get('verse_number', 'Unknown')} ({result2.get('confidence_score', 0)*100:.1f}%)")
    if "chapters_retrieved" in result2:
        print(f"Chapters: {result2['chapters_retrieved']}")

print("\n" + "="*80)
print("TEST 3: Out-of-Context Question (Should Reject)")
print("="*80)

query3 = "Who is Narendra Modi?"
print(f"\nQuery: {query3}\n")

result3 = chatbot.query(query3)

if "answer" in result3 and result3["answer"] == "Answer not found":
    print("\n[RESULT] PASS - CORRECTLY REJECTED - Answer not found")
else:
    print(f"\n[RESULT] FAIL - FALSE POSITIVE - Returned verse {result3.get('verse_number')}")

print("\n" + "="*80)
print("TESTING COMPLETE")
print("="*80)
