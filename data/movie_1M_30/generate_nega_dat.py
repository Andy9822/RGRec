import numpy as np

user_pos = {}
user_nega = {}
# ratings_np = np.loadtxt("./ratings_final.txt",dtype=np.int64)
# np.save("./ratings.npy", ratings_np)
# ratings_np = np.load("./ratings.npy")
train_np = np.load("./model/train.npy")
eval_np = np.load("./model/eval.npy")
test_np = np.load("./model/test.npy")
ratings_np = np.concatenate((train_np, eval_np, test_np), axis=0)
item_set = set(ratings_np[:, 1])

user_pos = {}
positive_ratings = ratings_np[np.where(ratings_np[:, 2] == 1)]

for one_rating in positive_ratings:
    u_id = one_rating[0]
    i_id = one_rating[1]
    if u_id not in user_pos:
        user_pos[u_id] = set()
    user_pos[u_id].add(i_id)

with open("./negative_data.txt",'w', encoding="UTF-8") as f:
    for idx, u_id in enumerate(user_pos.keys()):
        print("{}/{}\n".format(idx,len(user_pos.keys())))
        f.write("{}".format(u_id))
        pos_set = user_pos[u_id]
        unwatched_set = item_set - pos_set
        for item in np.random.choice(list(unwatched_set), size=100, replace=False):
            f.write('\t{}'.format(item))
        f.write("\n")
