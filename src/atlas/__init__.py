"""Atlas — multimodal agentic research system."""

import os
import sys

# faiss-cpu and PyTorch wheels for macOS each bundle their own OpenMP runtime
# (libomp). Loading both in one process aborts/segfaults unless duplicate
# runtimes are tolerated. Linux wheels (Docker/K8s) share one runtime, so this
# is scoped to macOS only. Must run before faiss/torch are imported.
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

__version__ = "0.1.0"
