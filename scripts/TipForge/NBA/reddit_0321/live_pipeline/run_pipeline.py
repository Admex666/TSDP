import os
import subprocess
import sys

def main():
    print("🚀 Starting Daily NBA Live Betting Pipeline...")
    pipeline_dir = r"E:\Data\TSDP\scripts\TipForge\NBA\reddit_0321\live_pipeline"
    
    scripts = [
        "live_odds.py",
        "live_features.py",
        "live_inference.py",
        "live_betting.py",
        "live_notify.py"
    ]
    
    for script in scripts:
        print(f"\n=========================================================")
        print(f"▶▶ Futtatás: {script}")
        print(f"=========================================================")
        
        script_path = os.path.join(pipeline_dir, script)
        result = subprocess.run([sys.executable, script_path], cwd=pipeline_dir)
        
        if result.returncode != 0:
            print(f"\n❌ KRITIKUS HIBA: a(z) {script} futtatása megszakadt. A Pipeline leállt.")
            sys.exit(1)
            
    print("\n✅ Minden modul lefutott! A telegram üzenet kiküldve.")

if __name__ == "__main__":
    main()
