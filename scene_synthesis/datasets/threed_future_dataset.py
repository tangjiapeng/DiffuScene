
import numpy as np
import pickle

from .utils import parse_threed_future_models
import torch
from torch.utils.data import Dataset, dataloader

class ThreedFutureDataset(object):
    def __init__(self, objects):
        assert len(objects) > 0
        self.objects = objects

    def __len__(self):
        return len(self.objects)

    def __str__(self):
        return "Dataset contains {} objects with {} discrete types".format(
            len(self)
        )

    def __getitem__(self, idx):
        return self.objects[idx]

    def _filter_objects_by_label(self, label):
        return [oi for oi in self.objects if oi.label == label]

    def get_closest_furniture_to_box(self, query_label, query_size):
        objects = self._filter_objects_by_label(query_label)

        mses = {}
        for i, oi in enumerate(objects):
            mses[oi] = np.sum((oi.size - query_size)**2, axis=-1)
        sorted_mses = [k for k, v in sorted(mses.items(), key=lambda x:x[1])]
        return sorted_mses[0]

    def get_closest_furniture_to_2dbox(self, query_label, query_size):
        objects = self._filter_objects_by_label(query_label)

        mses = {}
        for i, oi in enumerate(objects):
            mses[oi] = (
                (oi.size[0] - query_size[0])**2 +
                (oi.size[2] - query_size[1])**2
            )
        sorted_mses = [k for k, v in sorted(mses.items(), key=lambda x: x[1])]
        return sorted_mses[0]

    def get_closest_furniture_to_objfeats(self, query_label, query_objfeat):
        objects = self._filter_objects_by_label(query_label)

        mses = {}
        for i, oi in enumerate(objects):
            if query_objfeat.shape[0] == 32:
                mses[oi] = np.sum((oi.raw_model_norm_pc_lat32() - query_objfeat)**2, axis=-1)
            else:
                mses[oi] = np.sum((oi.raw_model_norm_pc_lat() - query_objfeat)**2, axis=-1)
        sorted_mses = [k for k, v in sorted(mses.items(), key=lambda x:x[1])]
        return sorted_mses[0]

    def get_closest_furniture_to_objfeats_and_size(self, query_label, query_objfeat, query_size):
        objects = self._filter_objects_by_label(query_label)

        objs = []
        mses_feat = []
        mses_size = []
        for i, oi in enumerate(objects):
            if query_objfeat.shape[0] == 32:
                mses_feat.append( np.sum((oi.raw_model_norm_pc_lat32() - query_objfeat)**2, axis=-1) )
            else:
                mses_feat.append( np.sum((oi.raw_model_norm_pc_lat() - query_objfeat)**2, axis=-1) )
            mses_size.append( np.sum((oi.size - query_size)**2, axis=-1) )
            objs.append(oi)

        ind = np.lexsort( (mses_feat, mses_size) )
        return objs[ ind[0] ]

    @classmethod
    def from_dataset_directory(
        cls, dataset_directory, path_to_model_info, path_to_models
    ):
        objects = parse_threed_future_models(
            dataset_directory, path_to_models, path_to_model_info
        )
        return cls(objects)

    @classmethod
    def from_pickled_dataset(cls, path_to_pickled_dataset):
        with open(path_to_pickled_dataset, "rb") as f:
            dataset = pickle.load(f)
        return dataset


class ThreedFutureNormPCDataset(ThreedFutureDataset):
    def __init__(self, objects, num_samples=2048):
        super().__init__(objects)

        self.num_samples = num_samples

    def __len__(self):
        return len(self.objects)

    def __str__(self):
        return "Dataset contains {} objects with {} discrete types".format(
            len(self)
        )

    def __getitem__(self, idx):
        obj = self.objects[idx]
        model_uid = obj.model_uid
        model_jid = obj.model_jid
        raw_model_path = obj.raw_model_path
        raw_model_norm_pc_path = obj.raw_model_norm_pc_path
        points = obj.raw_model_norm_pc()

        points_subsample = points[np.random.choice(points.shape[0], self.num_samples), :]

        points_torch = torch.from_numpy(points_subsample).float()
        data_dict =  {"points": points_torch, "idx": idx} 
        return data_dict

    def get_model_jid(self, idx):
        obj = self.objects[idx]
        model_uid = obj.model_uid
        model_jid = obj.model_jid
        data_dict =  {"model_jid": model_jid} 
        return data_dict

    def collate_fn(self, samples):
        ''' Collater that puts each data field into a tensor with outer dimension
            batch size.
        Args:
            samples: samples
        '''
    
        samples = list(filter(lambda x: x is not None, samples))
        return dataloader.default_collate(samples)


