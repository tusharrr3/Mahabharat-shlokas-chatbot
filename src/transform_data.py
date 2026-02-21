"""
Data transformation script to convert output.json to the expected format
"""

import json
import re
import sys
import os

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import DATA_DIR


def extract_verse_number(entry):
    """Extract verse number from the entry."""
    # Try to find it in the translation field that starts with "BG"
    for key, value in entry.items():
        if isinstance(key, str) and key.startswith("BG "):
            # Extract verse number from "BG 10.4-5:" format
            match = re.search(r'BG\s+([\d\.]+(?:-\d+)?)', key)
            if match:
                return match.group(1)
    
    # If not found, try the first key with long value (likely contains verse number)
    for key, value in entry.items():
        if isinstance(value, str) and value.isdigit() and len(value) <= 4:
            # This is likely the verse position, not the number
            continue
    
    return "Unknown"


def transform_entry(entry, index):
    """Transform a single entry to the expected format."""
    try:
        # Get the null array which contains the main data
        null_array = entry.get("null", [])
        
        if not null_array or len(null_array) < 5:
            print(f"Warning: Entry {index} has invalid null array")
            return None
        
        # Extract components from null array
        devanagari_shlok = null_array[0] if len(null_array) > 0 else ""
        transliterated_shlok = null_array[1] if len(null_array) > 1 else ""
        word_meanings = null_array[2] if len(null_array) > 2 else ""
        english_transliteration = null_array[3] if len(null_array) > 3 else ""
        translation = null_array[4] if len(null_array) > 4 else ""
        commentary = null_array[5] if len(null_array) > 5 else ""
        
        # Extract verse number from translation (e.g., "BG 10.4-5:...")
        verse_number = "Unknown"
        if translation and translation.startswith("BG "):
            match = re.search(r'BG\s+([\d\.]+(?:-\d+)?)', translation)
            if match:
                verse_number = match.group(1)
            # Remove the "BG X.Y:" prefix from translation
            translation = re.sub(r'^BG\s+[\d\.]+(?:-\d+)?:\s*', '', translation)
        
        # Combine meaning: word meanings + commentary
        meaning = ""
        if word_meanings:
            meaning += f"Word Meanings: {word_meanings}\n\n"
        if commentary:
            meaning += f"Commentary: {commentary}"
        
        # Create the transformed entry
        transformed = {
            "verse_number": verse_number,
            "shlok": devanagari_shlok,
            "meaning": meaning.strip(),
            "translation": translation
        }
        
        return transformed
        
    except Exception as e:
        print(f"Error transforming entry {index}: {e}")
        return None


def transform_json(input_path, output_path):
    """Transform output.json to mahabharat.json format."""
    try:
        print(f"Loading data from {input_path}...")
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Found {len(data)} entries")
        
        transformed_data = []
        skipped = 0
        
        for i, entry in enumerate(data):
            transformed = transform_entry(entry, i)
            if transformed:
                transformed_data.append(transformed)
            else:
                skipped += 1
        
        print(f"Successfully transformed {len(transformed_data)} entries")
        if skipped > 0:
            print(f"Skipped {skipped} invalid entries")
        
        # Save transformed data
        print(f"Saving transformed data to {output_path}...")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(transformed_data, f, ensure_ascii=False, indent=2)
        
        print("Transformation complete!")
        print(f"Output saved to: {output_path}")
        
        # Display sample
        if transformed_data:
            print("\nSample transformed entry:")
            print(json.dumps(transformed_data[0], ensure_ascii=False, indent=2))
        
        return True
        
    except Exception as e:
        print(f"Error during transformation: {e}")
        return False


def main():
    """Main function."""
    input_path = os.path.join(DATA_DIR, "output.json")
    output_path = os.path.join(DATA_DIR, "mahabharat.json")
    
    print("="*80)
    print("DATA TRANSFORMATION TOOL")
    print("="*80)
    print(f"\nInput:  {input_path}")
    print(f"Output: {output_path}")
    print()
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        return
    
    # Check if output file already exists
    if os.path.exists(output_path):
        response = input(f"\n{output_path} already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Transformation cancelled.")
            return
    
    # Perform transformation
    success = transform_json(input_path, output_path)
    
    if success:
        print("\n✓ Transformation successful!")
        print(f"\nYou can now run the chatbot with: python src/main.py")
    else:
        print("\n✗ Transformation failed!")


if __name__ == "__main__":
    main()
