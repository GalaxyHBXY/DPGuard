from DPGuard import *

detector = DPGuard(binary_model_path="binary_model/binary_rn101_ep6.pth", mllm_model="gpt-4o")

# No DP Demo
out = detector.detect("demo_img/nodp_demo.jpg")
print(out)

print()

# DP Demo
out = detector.detect("demo_img/dp_demo.jpg")
print(out)