from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from word2vec_pytorch_pipeline import run_pipeline


if __name__ == "__main__":
    run_pipeline(
        model_type="cbow",
        project_title="Project 1: CBOW PyTorch Evaluation Results",
        output_vec_name="cbow.vec",
        project_dir=Path(__file__).resolve().parent,
    )
