import sys
from pathlib import Path
# support running without installing as a package
import torch

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import time
from tqdm import tqdm
from lit_llama import Tokenizer

global_tokenizer = Tokenizer(Path("../checkpoints/lit-llama/tokenizer.model"))


def getUserTraj(path):
    file = open(path, encoding="Latin-1")
    print("opened successfully, reading lines...")
    line = file.readline()
    count = 0
    poi_id_list = []
    poi_info_list = []
    user_list = []
    user_traj_list = []
    # spilt user traj from the large file
    #while line and count <= 10000:
    while line:
        splited_line = line.split('\t')
        user_id = splited_line[0]
        poi_id = splited_line[1]
        cate_name = splited_line[3]
        lon = float(splited_line[4])
        lat = float(splited_line[5])
        time_offset = int(splited_line[-2])
        tm = time.strptime(splited_line[-1], "%a %b %d %H:%M:%S +0000 %Y\n")
        tm = time.localtime(time.mktime(tm) + time_offset*60)  # cal an offset to the UTC time

        if poi_id not in poi_id_list:
            poi_id_list.append(poi_id)
            tokened_name = global_tokenizer.encode(cate_name, bos=False, eos=False)
            poi_info_list.append((len(poi_info_list), cate_name, tokened_name.size(0), lon, lat))
        tuple = poi_info_list[poi_id_list.index(poi_id)] + (tm,)

        if user_id not in user_list:
            user_list.append(user_id)
            user_traj_list.append([tuple])
        else:
            user_traj_list[user_list.index(user_id)].append(tuple)

        line = file.readline()
        count += 1
    print("read in {} lines, get {} pois and {} trajectories.".format(count,
                                                                      len(poi_info_list),
                                                                      len(user_traj_list)))
    file.close()
    return poi_info_list, user_traj_list


def spatialEncode(coord_seq, method="simple", prompt_len=0):
    spatial_info = []
    for _ in range(prompt_len):
        spatial_info.append(torch.tensor([0.0, 0.0]))
    if method == "simple":
        for coord in coord_seq:
            spatial_info.append(torch.tensor(coord))

    return spatial_info


def splitSequence(user_traj_list, legal_len, min_len=5):
    # user_traj_list[0]->[(poi_id, cat_name, cat_token_len, lon, lat, time),...]
    traj_list = []
    for traj in tqdm(user_traj_list):
        splited_traj = []
        length = 0
        for place in traj:
            splited_traj.append(place)
            length += place[2]
            if length >= legal_len:
                traj_list.append(splited_traj)
                splited_traj = []
                length = 0
        if len(splited_traj) >= min_len:
            traj_list.append(splited_traj)

    return traj_list


def makeSequence(user_traj_list, add_prompt=True, split=0.0, max_len=256):
    # user_traj_list[0]->[(poi_id, cat_name, cat_token_len, lon, lat, time),...]
    prompt_head = "Given a user's visited places sequence as follows, " \
                  "predict which place the user will visit next: "
    promt_len = len(global_tokenizer.encode(prompt_head, bos=False, eos=False))

    # split over length sequence
    traj_list = splitSequence(user_traj_list, legal_len=(max_len-promt_len)//2, min_len=5)
    # traj_list[0]->[(poi_id, cat_name, cat_token_len, lon, lat, time), ...]
    prepared_sentences = []
    max_length = 0
    for traj in tqdm(traj_list):
        places = []  # places category names
        poi_token = []
        coord_seq = []  # places coordinates
        for place in traj:
            poi_token.append(place[0])
            places.append(place[1])
            coord_seq.append((0.0, 0.0))
            for _ in range(place[2]):
                coord_seq.append((place[3], place[4]))
        places_sequence = " , ".join(places)  # generate places sequence as a sentence
        spatial_addition = spatialEncode(coord_seq, method="simple", prompt_len=promt_len-1)

        if add_prompt:
            sentence = prompt_head + places_sequence
        else:
            sentence = places_sequence
        sentence_token = global_tokenizer.encode(sentence, bos=True, eos=False)

        assert len(sentence_token) == len(spatial_addition)

        prepared = {"sentence": sentence, # string
                    "sentence_token": sentence_token, # input token
                    "poi_token": poi_token, # poi id
                    "spatial_addition": spatial_addition, # input traj lat lon
                    "prompt_len": promt_len-1}
        length = len(sentence_token)
        if length > max_length:
            max_length = length
        prepared_sentences.append(prepared)

    print("max_length: {}".format(max_length))
    if not split:
        return prepared_sentences
    else:
        train_set = []
        test_set = []
        for i in range(len(prepared_sentences)):
            if i <= split*len(prepared_sentences):
                train_set.append(prepared_sentences[i])
            else:
                test_set.append(prepared_sentences[i])
        return train_set, test_set


if __name__ == "__main__":
    poi_info_list, user_traj_list = getUserTraj("../data/dataset_tsmc2014/dataset_TSMC2014_NYC.txt")
    train_set, test_set = makeSequence(user_traj_list, add_prompt=True, split=0.8, max_len=256)
    torch.save(train_set, "../data/spatial_dataset/poi_train.pt")
    torch.save(test_set, "../data/spatial_dataset/poi_test.pt")
    torch.save(poi_info_list, "../data/spatial_dataset/poi_list.pt")
    print("Completed!")

