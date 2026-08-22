"""
Read-only inspection of a checkpoint's classifier head architecture.
Doesn't load the model, doesn't modify anything -- just prints the shapes.

Usage:
    python inspect_checkpoint.py models/specialists/brain_tumor_specialist_v1.pt
"""
import sys
import torch

def inspect(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        print("Checkpoint is wrapped in a dict with keys:", list(checkpoint.keys()))
        if "class_names" in checkpoint:
            print("class_names found:", checkpoint["class_names"])
        else:
            print("No 'class_names' key in checkpoint.")
    else:
        state_dict = checkpoint
        print("Checkpoint is a raw state_dict (no wrapper).")

    print(f"\nAll top-level module prefixes:")
    prefixes = sorted(set(k.split(".")[0] for k in state_dict.keys()))
    print(" ", prefixes)

    print(f"\nFull classifier.* keys and shapes, in order:")
    classifier_keys = [(k, tuple(v.shape)) for k, v in state_dict.items() if k.startswith("classifier.")]
    for k, shape in classifier_keys:
        print(f"  {k:<30} {shape}")

    if not classifier_keys:
        print("  (none found -- head may use a different prefix, e.g. 'fc.' -- showing all keys instead:)")
        for k, v in state_dict.items():
            print(f"  {k:<30} {tuple(v.shape)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_checkpoint.py <path_to_checkpoint.pt>")
        sys.exit(1)
    inspect(sys.argv[1])
