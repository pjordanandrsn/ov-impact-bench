"""Fix-vs-bug litmus: does the P630 compile the LLM's GPU kernels?

On the stock 2026.1.0 wheel this aborts with cl::BuildError (clBuildProgram) —
the pre-fix __local block-read failure. With a build that includes PR #35712 it
should print GPU_COMPILE_OK. Uses only core OpenVINO (no GenAI), so it's
independent of openvino-genai version matching.
"""
import glob
import sys

import openvino as ov

print("OV build:", ov.get_version())
core = ov.Core()
print("devices:", core.available_devices)

xmls = glob.glob("/work/hf-cache/**/openvino_model.xml", recursive=True)
if not xmls:
    print("NO_MODEL_XML")
    sys.exit(2)
print("model:", xmls[0])

model = core.read_model(xmls[0])
core.compile_model(model, "GPU")  # <-- clBuildProgram happens here
print("GPU_COMPILE_OK — kernels built on P630")
