import torch
from torch import nn
import math


def gen_sentence(instruction, poi_token, poi_list, max_poi_len, tokenizer, device, max_token_len):
    places_id = poi_token
    encoded_prompt = tokenizer.encode(instruction, bos=True, eos=False, device=device)
    encoded_split = tokenizer.encode(',', bos=False, eos=False)
    encoded_gap = tokenizer.encode(" ", bos=False, eos=False)  # jn: abandon gap ^^
    lang_input = encoded_prompt
    patched_pos_input = torch.zeros((len(lang_input), 2), device=device, dtype=torch.float32)
    raw_pos_input = []
    spatial_scopes = []
    # pos_idx = [len(lang_input)]
    # print(f"start at {pos_start_idx}, patch_len = {max_poi_len}+{len(encoded_split)}")
    num_poi = 0
    for place_id in places_id:
        num_poi += 1
        place = poi_list[place_id][1]
        lon, lat = poi_list[place_id][4], poi_list[place_id][3]  # We use [lon,lat] while the dataset used lat,lon

        encoded_place = tokenizer.encode(place, bos=False, eos=False)

        # # jn: abandon gap ^^
        # gap_len = max_poi_len - encoded_place.size(0)
        # for _ in range(gap_len):
        #     encoded_place = torch.cat([encoded_place, encoded_gap])

        encoded_place = torch.cat([encoded_place, encoded_split])

        # # jn: abandon gap ^^
        # assert len(encoded_place) == len(encoded_split) + max_poi_len

        spatial_scopes.append([len(lang_input), len(lang_input)+len(encoded_place)])  # *1*
        raw_pos_input.append([lon, lat])

        lang_input = torch.cat([lang_input, encoded_place])  # after *1*
        pos_tensor = torch.tensor([[lon, lat]], device=device, dtype=torch.float32)
        patched_pos_input = torch.cat([patched_pos_input, pos_tensor.repeat(len(encoded_place), 1)])
        if max_token_len < len(lang_input) + len(encoded_split) + max_poi_len:
            break
        # if idx < len(places_id) - 1:
        #     pos_idx.append(len(lang_input))
    spatial_scopes = torch.tensor(spatial_scopes, device=device, dtype=torch.float32)
    raw_pos_input = torch.tensor(raw_pos_input, device=device, dtype=torch.float32)

    return lang_input, patched_pos_input, raw_pos_input, spatial_scopes, num_poi


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


def prepare(datas: list, poi_list: list, tokenizer, padded_vocab_size=None):
    assert padded_vocab_size is not None
    # datas = [[(poi_id, cat, token_len, lon, lat, timestamp),...]]
    # poi_list = [(poi_id, cat, token_len, lon, lat),...]
    max_poi_len = 0
    instruction = "Given a user's visited places sequence as follows, " \
                  "predict which place the user will visit next: "
    bbox = get_bbox(poi_list)  # boundary box
    for item in poi_list:
        max_poi_len = max(max_poi_len, item[2])
    data_ret = []
    encoded_split = tokenizer.encode(',', bos=False, eos=False)
    encoded_prompt = tokenizer.encode(instruction, bos=True, eos=False, device="cpu")
    patch_len = max_poi_len + len(encoded_split)
    pos_start_idx = len(encoded_prompt)
    for i, data in enumerate(datas):
        sentence = data["sentence"]  # language
        # print("correct answer: \n{}".format(sentence))
        # spatial_addition = data["spatial_addition"]
        poi_seq = data["poi_token"]  # poi_id_list

        prompt_sentence, spatial_addition, raw_spatial_addition, \
            spatial_scopes, num_poi = gen_sentence(instruction, poi_seq, poi_list,
                                                   max_poi_len, tokenizer, 'cpu', 256)
        spatial_addition = norm_spatial_seq(spatial_addition, bbox)
        raw_spatial_addition = norm_spatial_seq(raw_spatial_addition, bbox)

        legal_poi_seq = poi_seq[:num_poi]
        sample = {}
        sample['language_inputs'] = prompt_sentence[:-1]
        sample['language_labels'] = prompt_sentence[1:]
        # sample['language_labels_onehot'] = torch.nn.functional.one_hot(sample['language_labels'],
        #                                                                num_classes=padded_vocab_size).to(torch.float32)
        sample['spatial_inputs'] = spatial_addition[:-1]
        sample['spatial_labels'] = spatial_addition[1:]
        sample['raw_spatial'] = raw_spatial_addition
        sample['raw_spatial_labels'] = raw_spatial_addition[1:]
        sample['spatial_masks'] = torch.all(raw_spatial_addition[1:] == torch.tensor([0.0, 0.0]), dim=1)

        trainable_mask = torch.zeros(sample['language_inputs'].shape[0], dtype=torch.bool)
        # trainable_mask[:sample_len] = True
        sample['trainable_masks'] = trainable_mask
        sample['infer_poi'] = [poi_list[i] for i in legal_poi_seq]
        spatial_scopes[-1][1] -= 1  # jn: do this because spatial_inputs remove the last node
        sample['spatial_scopes'] = spatial_scopes  # jn: include starts and ends (in case we need multiple length)
        # sample['spatial_start_idx'] = get_spatial_idx()  # jn:abandon this, see "spatial_scopes"
        # sample['pos_start_idx'] = pos_start_idx
        # sample['patch_len'] = patch_len
        sample['poi_num'] = num_poi
        data_ret.append(sample)
        # optionally crop the logits to only the top k options

    return data_ret, (pos_start_idx, patch_len, bbox)


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
