from utils.util.loader import load2json, get_data_folder, natural_sort_key

class PersonMap:
    def __init__(self, dataset_name: str, filename = 'person_map.json'):
        path = get_data_folder( dataset_name=dataset_name, path="")
        self.person2id = load2json(filepath = path+filename)
        ## 兼容旧数据集
        self.person2id = {v: v for _, v in self.person2id.items()} if dataset_name in ["CYBHi", "heartprint"] else self.person2id
        ##
        self.id2person = {v: k for k, v in self.person2id.items()}
    
    def get_id(self, person_id):
        if isinstance(person_id, int):
            return person_id
        return self.person2id[person_id] if person_id in self.person2id else -1
    
    def get_person(self, person_id):
        return self.id2person[person_id] if person_id in self.id2person else "None"
    
    def get_new_person2id(self, person_ids):
        person_names = [self.get_person(id) for id in person_ids]
        unique_persons = sorted(list(set(person_names)), key=natural_sort_key)
        new_person_map = {person: idx for idx, person in enumerate(unique_persons)}
        return new_person_map