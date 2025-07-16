#!/usr/bin/env python3
"""
Check downloaded model structure and files
"""
import os
from pathlib import Path

def check_model_structure():
    models_base_dir = Path("./models")
    
    print("🔍 Checking model structure...")
    print("=" * 50)
    
    # Check STT model
    stt_dir = models_base_dir / "stt" / "models--kyutai--stt-1b-en_fr"
    if stt_dir.exists():
        print(f"✅ STT model directory exists: {stt_dir}")
        snapshot_path = next(stt_dir.glob("**/snapshots/*/"), None)
        if snapshot_path:
            print(f"✅ STT snapshot path: {snapshot_path}")
            print("STT files:")
            for file in snapshot_path.iterdir():
                print(f"  - {file.name}")
        else:
            print("❌ STT snapshot path not found")
    else:
        print("❌ STT model directory not found")
    
    print()
    
    # Check TTS model
    tts_dir = models_base_dir / "tts" / "models--kyutai--tts-1.6b-en_fr"
    if tts_dir.exists():
        print(f"✅ TTS model directory exists: {tts_dir}")
        snapshot_path = next(tts_dir.glob("**/snapshots/*/"), None)
        if snapshot_path:
            print(f"✅ TTS snapshot path: {snapshot_path}")
            print("TTS files:")
            for file in snapshot_path.iterdir():
                print(f"  - {file.name}")
        else:
            print("❌ TTS snapshot path not found")
    else:
        print("❌ TTS model directory not found")
    
    print()
    
    # Check LLM model
    llm_dir = models_base_dir / "llm" / "models--kyutai--moshika-pytorch-bf16"
    if llm_dir.exists():
        print(f"✅ LLM model directory exists: {llm_dir}")
        snapshot_path = next(llm_dir.glob("**/snapshots/*/"), None)
        if snapshot_path:
            print(f"✅ LLM snapshot path: {snapshot_path}")
            print("LLM files:")
            for file in snapshot_path.iterdir():
                print(f"  - {file.name}")
        else:
            print("❌ LLM snapshot path not found")
    else:
        print("❌ LLM model directory not found")
    
    print("=" * 50)

if __name__ == "__main__":
    check_model_structure()