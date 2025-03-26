from DPGuard import *

detector_gpt = DPGuard(binary_model_path="binary_model/binary_rn101_ep6.pth", mllm_model="gpt-4o")
detector_gemini = DPGuard(binary_model_path="binary_model/binary_rn101_ep6.pth", mllm_model="gemini-2.5-pro-exp-03-25")

# No DP Demo
out = detector_gpt.detect("demo_img/nodp_demo.jpg")
print(out)

print()

# DP Demo
out = detector_gemini.detect("demo_img/dp_demo.jpg")
print(out)