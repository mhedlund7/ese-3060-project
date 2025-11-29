import torch

# pick log to inspect
log = torch.load("logs/ca3c8473-3858-4db6-b129-7003b4aaf1e4/log.pt", map_location="cpu")

print(log.keys())

accs = log["accs"]
times = log["times"]
timestamp = log["timestamp"]
compile = log["compile"]
print("Accs:", accs.tolist())
print("Mean:", accs.mean().item())
print("Std:", accs.std().item())
print("Times:", times.tolist())
print("Mean time:", times.mean().item())
print("Std time:", times.std().item())
print("Timestamp:", timestamp)
print("Compile:", compile)
# To see saved code:
# print(log["code"])