"""Test exact verse lookup with chunked documents"""

import os
os.environ["GEMINI_API_KEY"] = "AIzaSyDuXWI8DGWo9ejCO8xt0DaDGHAU_LSgbgw"

import sys
sys.path.insert(0, "src")

from main import MahabharatChatbot

print("=" * 80)
print("TESTING EXACT VERSE LOOKUP WITH CHUNKING")
print("=" * 80)

chatbot = MahabharatChatbot()

# Test 1: Chapter 2 Verse 47 (the one user reported)
print("\n" + "=" * 80)
print("TEST 1: chapter 2 verse 47")
print("=" * 80)

result1 = chatbot.query("chapter 2 verse 47")

if "verse_number" in result1:
    print(f"PASS - FOUND: Verse {result1['verse_number']}")
    print(f"  Confidence: {result1.get('confidence_score', 0) * 100:.1f}%")
    print(f"  Chapter: {result1.get('chapter', 'N/A')}")
else:
    print(f"FAIL - NOT FOUND: {result1.get('answer', 'Unknown error')}")

# Test 2: Chapter 5 Verse 10 (should also work)
print("\n" + "=" * 80)
print("TEST 2: chapter 5 verse 10")
print("=" * 80)

result2 = chatbot.query("chapter 5 verse 10")

if "verse_number" in result2:
    print(f"PASS - FOUND: Verse {result2['verse_number']}")
    print(f"  Confidence: {result2.get('confidence_score', 0) * 100:.1f}%")
    print(f"  Chapter: {result2.get('chapter', 'N/A')}")
else:
    print(f"FAIL - NOT FOUND: {result2.get('answer', 'Unknown error')}")

# Test 3: Verse in middle of chunk (e.g., 2.48 if chunk is 2.47-49)
print("\n" + "=" * 80)
print("TEST 3: chapter 2 verse 48 (middle of chunk)")
print("=" * 80)

result3 = chatbot.query("chapter 2 verse 48")

if "verse_number" in result3:
    print(f"PASS - FOUND: Verse {result3['verse_number']}")
    print(f"  Confidence: {result3.get('confidence_score', 0) * 100:.1f}%")
    metadata = result3
    if 'verses_in_chunk' in str(result3):
        print(f"  Chunked document detected")
else:
    print(f"FAIL - NOT FOUND: {result3.get('answer', 'Unknown error')}")

print("\n" + "=" * 80)
print("TESTING COMPLETE")
print("=" * 80)
