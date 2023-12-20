import torch
from torch import nn
import math


instructions = [
    # instruction 0 :
    {"instruct_head":
        "Given the sequence of POIs that a user accesses, predict the next most likely POI name. "
        "Make sure the output is a POI name and not any other form. "
        "Avoid nonsense and irrelevant information. The sequence is as follows: \"",
     "instruct_tail": "\". ",
     "answer_head": "The next most likely POI name is: \"",
     "answer_tail": "\"."}
]


def gen_sentence(instruction, poi_token, poi_list, max_poi_len, tokenizer, max_token_len, device):
    places_id = poi_token
    instruct_head = tokenizer.encode(instruction["instruct_head"], bos=True, eos=False, device=device)
    instruct_tail = tokenizer.encode(instruction["instruct_tail"], bos=False, eos=False, device=device)
    answer_head = tokenizer.encode(instruction["answer_head"], bos=False, eos=False, device=device)
    answer_tail = tokenizer.encode(instruction["answer_tail"], bos=False, eos=True, device=device)
    encoded_split = tokenizer.encode(',', bos=False, eos=False)
    lang_input = instruct_head
    redundant_length = len(instruct_tail) + len(answer_head) + len(answer_tail)  # for total length calculation
    poi_mask = torch.zeros(len(lang_input), device=device, dtype=torch.bool)  # True for POIs names, False for the others
    raw_pos_input = []
    spatial_scopes = []
    poi_num = 0
    encoded_place = None
    lon, lat = 0.0, 0.0
    for i in range(len(places_id)-1):
        place_id = places_id[i]
        place = poi_list[place_id][1]
        lon, lat = poi_list[place_id][4], poi_list[place_id][3]  # We use [lon,lat] while the dataset used lat,lon

        encoded_place = tokenizer.encode(place, bos=False, eos=False)
        if max_token_len < len(lang_input) + len(encoded_place) + redundant_length + max_poi_len \
                or i >= len(places_id)-2:
            # if too long then stop here
            break

        if poi_num >= 1:  # add a split in front (to avoid redundant split at the end)
            lang_input = torch.cat([lang_input, encoded_split])
            poi_mask = torch.cat([poi_mask, torch.zeros(len(encoded_split), device=device, dtype=torch.bool)])

        spatial_scopes.append([len(lang_input), len(lang_input) + len(encoded_place)])  # now there is no ','
        raw_pos_input.append([lon, lat])
        lang_input = torch.cat([lang_input, encoded_place])
        poi_mask = torch.cat([poi_mask, torch.ones(len(encoded_place), device=device, dtype=torch.bool)])
        poi_num += 1

    lang_input = torch.cat([lang_input, instruct_tail])  # end Question
    poi_mask = torch.cat([poi_mask, torch.zeros(len(instruct_tail), device=device, dtype=torch.bool)])
    lang_input = torch.cat([lang_input, answer_head])  # start Answer
    poi_mask = torch.cat([poi_mask, torch.zeros(len(answer_head), device=device, dtype=torch.bool)])
    # add next POI answer
    spatial_scopes.append([len(lang_input), len(lang_input) + len(encoded_place)])
    raw_pos_input.append([lon, lat])
    lang_input = torch.cat([lang_input, encoded_place])
    poi_mask = torch.cat([poi_mask, torch.ones(len(encoded_place), device=device, dtype=torch.bool)])
    poi_num += 1

    lang_input = torch.cat([lang_input, answer_tail])  # end Answer
    poi_mask = torch.cat([poi_mask, torch.zeros(len(answer_tail), device=device, dtype=torch.bool)])

    spatial_scopes = torch.tensor(spatial_scopes, device=device, dtype=torch.float32)
    raw_pos_input = torch.tensor(raw_pos_input, device=device, dtype=torch.float32)

    return lang_input, poi_mask, raw_pos_input, spatial_scopes, poi_num


def get_bbox(poi_list):
    bbox = {'lon_min': 999, "lon_max": -999, 'lat_min': 999, "lat_max": -999}  # (boundary box) for readability
    for item in poi_list:
        if item[4] < bbox['lon_min']:
            bbox['lon_min'] = item[4]
        if item[4] > bbox['lon_max']:
            bbox['lon_max'] = item[4]
        if item[3] < bbox['lat_min']:
            bbox['lat_min'] = item[3]
        if item[3] > bbox['lat_max']:
            bbox['lat_max'] = item[3]
    return bbox


def prepare(datas: list, poi_list: list, tokenizer, padded_vocab_size=None, max_seq_length=256, stage="train"):
    assert padded_vocab_size is not None
    # datas = [[(poi_id, cat, token_len, lon, lat, timestamp),...]]
    # poi_list = [(poi_id, cat, token_len, lon, lat),...]
    max_poi_len = 0
    instruction = instructions[0]
    bbox = get_bbox(poi_list)  # boundary box
    for item in poi_list:
        max_poi_len = max(max_poi_len, item[2])
    data_ret = []
    # encoded_split = tokenizer.encode(',', bos=False, eos=False)
    # encoded_prompt = tokenizer.encode(instruction, bos=True, eos=False, device="cpu")
    # patch_len = max_poi_len + len(encoded_split)
    # pos_start_idx = len(encoded_prompt)
    for i, data in enumerate(datas):
        # sentence = data["sentence"]  # language
        # print("correct answer: \n{}".format(sentence))
        # spatial_addition = data["spatial_addition"]
        poi_seq = data["poi_token"]  # poi_id_list
        prompt_sentence, poi_mask, raw_spatial_addition, \
            spatial_scopes, num_poi = gen_sentence(instruction, poi_seq, poi_list,
                                                   max_poi_len, tokenizer, max_seq_length, 'cpu')
        if stage == "train" or stage == "valid":
            language_input = prompt_sentence[:-1]
            language_label = prompt_sentence[1:]
            poi_mask = poi_mask[:-1]
        else:  # if it's generation process, then get rid of the answer POI
            quotation = tokenizer.encode("\"", bos=False, eos=False)
            last_index = torch.where(prompt_sentence == quotation)[0][-1]
            language_input = prompt_sentence[:last_index+1]
            language_label = prompt_sentence
            poi_mask = poi_mask[:last_index+1]
            raw_spatial_addition = raw_spatial_addition[:-1]
            spatial_scopes = spatial_scopes[:-1]
            num_poi -= 1

        raw_spatial_addition = norm_spatial_seq(raw_spatial_addition, bbox)
        legal_poi_seq = poi_seq[:num_poi]
        sample = {}
        sample['language_inputs'] = language_input
        sample['language_labels'] = language_label
        # sample['language_labels_onehot'] = torch.nn.functional.one_hot(sample['language_labels'],
        #                                                                num_classes=padded_vocab_size).to(torch.float32)
        sample['poi_mask'] = poi_mask
        sample['raw_spatial'] = raw_spatial_addition
        sample['raw_spatial_labels'] = raw_spatial_addition[1:]
        sample['spatial_masks'] = torch.all(raw_spatial_addition[1:] == torch.tensor([0.0, 0.0]), dim=1)

        trainable_mask = torch.zeros(sample['language_inputs'].shape[0], dtype=torch.bool)
        # trainable_mask[:sample_len] = True
        sample['trainable_masks'] = trainable_mask
        sample['infer_poi'] = [poi_list[i] for i in legal_poi_seq]
        sample['spatial_scopes'] = spatial_scopes  # jn: include starts and ends (in case we need multiple length)
        # sample['spatial_start_idx'] = get_spatial_idx()  # jn:abandon this, see "spatial_scopes"
        # sample['pos_start_idx'] = pos_start_idx
        # sample['patch_len'] = patch_len
        sample['poi_num'] = num_poi
        data_ret.append(sample)
        # optionally crop the logits to only the top k options

    return data_ret, (max_poi_len, bbox)


def norm_spatial_seq(spatial_seq: torch.Tensor, bbox):
    # normalize loc
    min_corner = torch.tensor([[bbox['lon_min'], bbox['lat_min']]])
    max_corner = torch.tensor([[bbox['lon_max'], bbox['lat_max']]])
    assert not ((max_corner - min_corner) == 0).any()  # there shouldn't be any zeros
    normed = (spatial_seq - min_corner) / (max_corner - min_corner)
    normed[spatial_seq == 0.0] = 0.0  # do not normalize paddings
    return normed


class POI_Find_Dict:
    def __init__(self, poi_list):  # (len(poi_info_list), cate_name, tokened_name.size(0), lon, lat)
        self.bbox = get_bbox(poi_list)
        self.cat_dict = {}
        for poi in poi_list:
            cat_name = poi[1]
            if cat_name not in self.cat_dict:
                self.cat_dict[cat_name] = [poi]
            else:
                self.cat_dict[cat_name].append(poi)

    def find_cat_pos(self, output_cat, output_lon, output_lat, target_poi, normalized=True):
        # normalized==True: if output is normalized in 0~1 space instead of geo-space
        if output_cat not in self.cat_dict:
            return -1
        poi_list = self.cat_dict[output_cat]
        if normalized:
            lon, lat = self.de_normalize(output_lon, output_lat)
        sorted_data = sorted(poi_list, key=lambda x: self.haversine(x[3], x[4], lon, lat))
        # 只保留排序后的 id
        sorted_ids = [item[0] for item in sorted_data]
        if target_poi not in sorted_ids:
            return -1
        else:
            return sorted_ids.index(target_poi)

    def de_normalize(self, lon, lat):  # reverse the normalizing process
        lon = lon * (self.bbox['lon_max'] - self.bbox['lon_min']) + self.bbox['lon_min']
        lat = lat * (self.bbox['lat_max'] - self.bbox['lat_min']) + self.bbox['lat_min']
        return lon, lat

    def haversine(self, lon1, lat1, lon2, lat2):
        # 将经纬度转换为弧度
        lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
        # Haversine 公式
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        # 地球半径（单位：公里）
        R = 6371.0
        # 计算距离
        distance = R * c
        return distance
