import subprocess

ret = subprocess.run(["python", "001_pred_to_action.py"], capture_output=True, text=True)
output = ret.stdout.strip().split("\n")
output = [s for s in output if s.startswith("**********")]

for s in output:
    print(s)
