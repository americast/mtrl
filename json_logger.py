import json
import torch
from torch.utils.tensorboard import SummaryWriter
import pudb

writer = SummaryWriter("tb_logs/IRL")
# writer = SummaryWriter("tb_logs/org")

print("")
f = open("logs/90f2497ff4cee27c0d30fbc66e6ba205f94808ba4ea16e057df58e73_issue_None_seed_1/train.log", "r")
# f = open("/home/ssinha97/mtrl_org/logs/90f2497ff4cee27c0d30fbc66e6ba205f94808ba4ea16e057df58e73_issue_None_seed_1/train.log", "r")

count = 0
while True:
    line = f.readline()
    if not line: break
    json_here = json.loads(line)
    writer.add_scalar('Loss/reward_0', json_here['episode_reward_env_index_0'], count)
    writer.add_scalar('Loss/reward_1', json_here['episode_reward_env_index_1'], count)
    writer.add_scalar('Loss/reward_2', json_here['episode_reward_env_index_2'], count)
    writer.add_scalar('Loss/reward_3', json_here['episode_reward_env_index_3'], count)
    writer.add_scalar('Loss/reward_4', json_here['episode_reward_env_index_4'], count)
    writer.add_scalar('Loss/reward_5', json_here['episode_reward_env_index_5'], count)
    writer.add_scalar('Loss/reward_6', json_here['episode_reward_env_index_6'], count)
    writer.add_scalar('Loss/reward_7', json_here['episode_reward_env_index_7'], count)
    writer.add_scalar('Loss/reward_8', json_here['episode_reward_env_index_8'], count)
    writer.add_scalar('Loss/reward_9', json_here['episode_reward_env_index_9'], count)
    count += 1
    print(count, end="\r")
f.close()
print("\n")