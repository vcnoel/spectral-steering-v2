"""
Unified Entrypoint for Spectral Steering V2
"""
import argparse
import sys
import subprocess
import os
from pathlib import Path

def setup_parser():
    parser = argparse.ArgumentParser(description="Spectral Steering V2 - Main Orchestrator")
    subparsers = parser.add_subparsers(dest="command", help="Execution Phases")
    
    # Phase 1
    p1 = subparsers.add_parser("phase1", help="Run Phase 1 Sycophancy Evaluation")
    
    # Capability Tax
    cap = subparsers.add_parser("capability", help="Run Capability Tax matrix evaluation")
    cap.add_argument("--models", nargs="*", default=["llama-3.1-8b"], help="List of models to evaluate from configs")
    
    # Eigenvector Mechanistic 
    eig = subparsers.add_parser("eigenvector", help="Run Eigenvector Cross-Task extraction")
    
    # Universal Arguments
    parser.add_argument("--use-vllm", action="store_true", help="Enable vLLM generation backend (requires Linux/WSL or compiled Windows wheels)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for text generations")
    
    return parser

def execute_script(script_name, *args):
    script_path = Path(__file__).parent / "scripts" / script_name
    if not script_path.exists():
        print(f"[ERROR] Script not found: {script_path}")
        return
    cmd = [sys.executable, str(script_path)] + list(args)
    print(f"\n>>> Running Command: {' '.join(cmd)}\n")
    subprocess.run(cmd)

def main():
    parser = setup_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(0)
        
    print(f"[INFO] Initializing Spectral Steering V2 CLI -> Command: {args.command}")
    
    if args.use_vllm:
        print("[INFO] Attempting to use vLLM for inference optimization...")
        os.environ["SPECTRAL_USE_VLLM"] = "1"
        
    os.environ["SPECTRAL_BATCH_SIZE"] = str(args.batch_size)

    try:
        if args.command == "phase1":
            execute_script("run_phase1.py")
        elif args.command == "capability":
            execute_script("run_capability_tax.py")
        elif args.command == "eigenvector":
            execute_script("run_eigenvector_analysis.py")
        else:
            print(f"[ERROR] Unknown command: {args.command}")
    except Exception as e:
        print(f"[FATAL] Workflow interrupted: {e}")

if __name__ == "__main__":
    main()
