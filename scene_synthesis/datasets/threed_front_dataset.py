
from math import ceil
import numpy as np

from functools import lru_cache
from scipy.ndimage import rotate

import torch
from torch.utils.data import Dataset, dataloader


class DatasetDecoratorBase(Dataset):
    """A base class that helps us implement decorators for ThreeDFront-like
    datasets."""
    def __init__(self, dataset):
        self._dataset = dataset

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        return self._dataset[idx]

    @property
    def bounds(self):
        return self._dataset.bounds

    @property
    def n_classes(self):
        return self._dataset.n_classes

    @property
    def class_labels(self):
        return self._dataset.class_labels

    @property
    def class_frequencies(self):
        return self._dataset.class_frequencies

    @property
    def n_object_types(self):
        return self._dataset.n_object_types

    @property
    def object_types(self):
        return self._dataset.object_types

    @property
    def feature_size(self):
        return self.bbox_dims + self.n_classes

    @property
    def bbox_dims(self):
        raise NotImplementedError()
    
    # compute max_length for diffusion models
    @property
    def max_length(self):
        return self._dataset.max_length 

    def post_process(self, s):
        return self._dataset.post_process(s)


class BoxOrderedDataset(DatasetDecoratorBase):
    def __init__(self, dataset, box_ordering=None):
        super().__init__(dataset)
        self.box_ordering = box_ordering

    @lru_cache(maxsize=16)
    def _get_boxes(self, scene_idx):
        scene = self._dataset[scene_idx]
        if self.box_ordering is None:
            return scene.bboxes
        elif self.box_ordering == "class_frequencies":
            return scene.ordered_bboxes_with_class_frequencies(
                self.class_frequencies
            )
        else:
            raise NotImplementedError()


class DataEncoder(BoxOrderedDataset):
    """DataEncoder is a wrapper for all datasets we have
    """
    @property
    def property_type(self):
        raise NotImplementedError()


class RoomLayoutEncoder(DataEncoder):
    @property
    def property_type(self):
        return "room_layout"

    def __getitem__(self, idx):
        """Implement the encoding for the room layout as images."""
        img = self._dataset[idx].room_mask[:, :, 0:1]
        return np.transpose(img, (2, 0, 1))

    @property
    def bbox_dims(self):
        return 0


class ClassLabelsEncoder(DataEncoder):
    """Implement the encoding for the class labels."""
    @property
    def property_type(self):
        return "class_labels"

    def __getitem__(self, idx):
        # Make a local copy of the class labels
        classes = self.class_labels

        # Get the scene
        boxes = self._get_boxes(idx)
        L = len(boxes)  # sequence length
        C = len(classes)  # number of classes
        class_labels = np.zeros((L, C), dtype=np.float32)
        for i, bs in enumerate(boxes):
            class_labels[i] = bs.one_hot_label(classes)
        return class_labels

    @property
    def bbox_dims(self):
        return 0


class TranslationEncoder(DataEncoder):
    @property
    def property_type(self):
        return "translations"

    def __getitem__(self, idx):
        # Get the scene
        scene = self._dataset[idx]
        boxes = self._get_boxes(idx)
        L = len(boxes)  # sequence length
        translations = np.zeros((L, 3), dtype=np.float32)
        for i, bs in enumerate(boxes):
            translations[i] = bs.centroid(-scene.centroid)
        return translations

    @property
    def bbox_dims(self):
        return 3


class SizeEncoder(DataEncoder):
    @property
    def property_type(self):
        return "sizes"

    def __getitem__(self, idx):
        # Get the scene
        boxes = self._get_boxes(idx)
        L = len(boxes)  # sequence length
        sizes = np.zeros((L, 3), dtype=np.float32)
        for i, bs in enumerate(boxes):
            sizes[i] = bs.size
        return sizes

    @property
    def bbox_dims(self):
        return 3


class AngleEncoder(DataEncoder):
    @property
    def property_type(self):
        return "angles"

    def __getitem__(self, idx):
        # Get the scene
        boxes = self._get_boxes(idx)
        # Get the rotation matrix for the current scene
        L = len(boxes)  # sequence length
        angles = np.zeros((L, 1), dtype=np.float32)
        for i, bs in enumerate(boxes):
            angles[i] = bs.z_angle
        return angles

    @property
    def bbox_dims(self):
        return 1

class ObjFeatEncoder(DataEncoder):
    @property
    def property_type(self):
        return "objfeats"

    def __getitem__(self, idx):
        # Get the scene
        boxes = self._get_boxes(idx)
        # Get the rotation matrix for the current scene
        L = len(boxes)  # sequence length
        C = len(boxes[0].raw_model_norm_pc_lat())
        latents = np.zeros((L, C), dtype=np.float32)
        for i, bs in enumerate(boxes):
            latents[i, :] = bs.raw_model_norm_pc_lat()
        return latents

    @property
    def bbox_dims(self):
        return 64

class ObjFeat32Encoder(DataEncoder):
    @property
    def property_type(self):
        return "objfeats_32"

    def __getitem__(self, idx):
        # Get the scene
        boxes = self._get_boxes(idx)
        # Get the rotation matrix for the current scene
        L = len(boxes)  # sequence length
        C = len(boxes[0].raw_model_norm_pc_lat32())
        latents = np.zeros((L, C), dtype=np.float32)
        for i, bs in enumerate(boxes):
            latents[i, :] = bs.raw_model_norm_pc_lat32()
        return latents

    @property
    def bbox_dims(self):
        return 32

class DatasetCollection(DatasetDecoratorBase):
    def __init__(self, *datasets):
        super().__init__(datasets[0])
        self._datasets = datasets

    @property
    def bbox_dims(self):
        return sum(d.bbox_dims for d in self._datasets)

    def __getitem__(self, idx):
        sample_params = {}
        for di in self._datasets:
            sample_params[di.property_type] = di[idx]
        return sample_params

    @staticmethod
    def collate_fn(samples):
        # We assume that all samples have the same set of keys
        key_set = set(samples[0].keys()) - set(["length"])
        # remove text _keys
        text_keys = set( ["description", "desc_emb"] )
        key_set = key_set - text_keys

        # Compute the max length of the sequences in the batch
        max_length = max(sample["length"] for sample in samples)

        # Assume that all inputs that are 3D or 1D do not need padding.
        # Otherwise, pad the first dimension.
        padding_keys = set(k for k in key_set if len(samples[0][k].shape) == 2)
        sample_params = {}
        sample_params.update({
            k: np.stack([sample[k] for sample in samples], axis=0)
            for k in (key_set-padding_keys)
        })

        sample_params.update({
            k: np.stack([
                np.vstack([
                    sample[k],
                    np.zeros((max_length-len(sample[k]), sample[k].shape[1]))
                ]) for sample in samples
            ], axis=0)
            for k in padding_keys
        })
        sample_params["lengths"] = np.array([
            sample["length"] for sample in samples
        ])

        if "description" in samples[0].keys():        
            sample_params["description"] = [ sample["description"] for sample in samples]
        
        if "desc_emb" in samples[0].keys():  
            sample_params["desc_emb"] = np.stack([sample["desc_emb"] for sample in samples], axis=0)

        # Make torch tensors from the numpy tensors
        torch_sample = {
            k: torch.from_numpy(sample_params[k]).float()
            for k in sample_params if k != "description"
        }

        torch_sample.update({
            k: torch_sample[k][:, None]
            for k in torch_sample.keys()
            if "_tr" in k
        })

        if "description" in samples[0].keys():    
            torch_sample["description"] = sample_params["description"]

        return torch_sample


class CachedDatasetCollection(DatasetCollection):
    def __init__(self, dataset):
        super().__init__(dataset)
        self._dataset = dataset

    def __getitem__(self, idx):
        return self._dataset.get_room_params(idx)

    @property
    def bbox_dims(self):
        return self._dataset.bbox_dims


class RotationAugmentation(DatasetDecoratorBase):
    def __init__(self, dataset, min_rad=0.174533, max_rad=5.06145, fixed=False):
        super().__init__(dataset)
        self._min_rad = min_rad
        self._max_rad = max_rad
        self._fixed   = fixed
        
    @staticmethod
    def rotation_matrix_around_y(theta):
        R = np.zeros((3, 3))
        R[0, 0] = np.cos(theta)
        R[0, 2] = -np.sin(theta)
        R[2, 0] = np.sin(theta)
        R[2, 2] = np.cos(theta)
        R[1, 1] = 1.
        return R

    @property
    def rot_angle(self):
        if np.random.rand() < 0.5:
            return np.random.uniform(self._min_rad, self._max_rad)
        else:
            return 0.0
    
    @property
    def fixed_rot_angle(self):
        if np.random.rand() < 0.25:
            return np.pi * 1.5
        elif np.random.rand() < 0.50:
            return np.pi
        elif np.random.rand() < 0.75:
            return np.pi * 0.5
        else:
            return 0.0

    def __getitem__(self, idx):
        # Get the rotation matrix for the current scene
        if self._fixed:
            rot_angle = self.fixed_rot_angle
        else:
            rot_angle = self.rot_angle
        R = RotationAugmentation.rotation_matrix_around_y(rot_angle)

        sample_params = self._dataset[idx]
        for k, v in sample_params.items():
            if k == "translations":
                sample_params[k] = v.dot(R)
            elif k == "angles":
                angle_min, angle_max = self.bounds["angles"]
                sample_params[k] = \
                    (v + rot_angle - angle_min) % (2 * np.pi) + angle_min
            elif k == "room_layout":
                # Fix the ordering of the channels because it was previously
                # changed
                img = np.transpose(v, (1, 2, 0))
                sample_params[k] = np.transpose(rotate(
                    img, rot_angle * 180 / np.pi, reshape=False
                ), (2, 0, 1))
        return sample_params



class Scale(DatasetDecoratorBase):
    @staticmethod
    def scale(x, minimum, maximum):
        X = x.astype(np.float32)
        X = np.clip(X, minimum, maximum)
        X = ((X - minimum) / (maximum - minimum))
        X = 2 * X - 1
        return X

    @staticmethod
    def descale(x, minimum, maximum):
        x = (x + 1) / 2
        x = x * (maximum - minimum) + minimum
        return x

    def __getitem__(self, idx):
        bounds = self.bounds
        sample_params = self._dataset[idx]
        for k, v in sample_params.items():
            if k == "objfeats" or k == "objfeats_32":
                continue

            elif k in bounds:
                sample_params[k] = Scale.scale(
                    v, bounds[k][0], bounds[k][1]
                )
        return sample_params

    def post_process(self, s):
        bounds = self.bounds
        sample_params = {}
        for k, v in s.items():
            if k == "room_layout" or k == "class_labels" or k == "relations" or k == "description" or k == "desc_emb":
                sample_params[k] = v
                
            elif k == "objfeats" or k == "objfeats_32":
                continue
            
            else:
                sample_params[k] = Scale.descale(
                    v, bounds[k][0], bounds[k][1]
                )
        return super().post_process(sample_params)

    @property
    def bbox_dims(self):
        return 3 + 3 + 1


class Scale_CosinAngle(DatasetDecoratorBase):
    @staticmethod
    def scale(x, minimum, maximum):
        X = x.astype(np.float32)
        X = np.clip(X, minimum, maximum)
        X = ((X - minimum) / (maximum - minimum))
        X = 2 * X - 1
        return X

    @staticmethod
    def descale(x, minimum, maximum):
        x = (x + 1) / 2
        x = x * (maximum - minimum) + minimum
        return x

    def __getitem__(self, idx):
        bounds = self.bounds
        sample_params = self._dataset[idx]
        for k, v in sample_params.items():
            if k == "angles":
                # [cos, sin]
                sample_params[k] = np.concatenate([np.cos(v), np.sin(v)], axis=-1)

            elif k == "objfeats" or k == "objfeats_32":
                continue
            
            elif k in bounds:
                sample_params[k] = Scale.scale(
                    v, bounds[k][0], bounds[k][1]
                )
        return sample_params

    def post_process(self, s):
        bounds = self.bounds
        sample_params = {}
        for k, v in s.items():
            if k == "room_layout" or k == "class_labels" or k == "relations" or k == "description" or k == "desc_emb":
                sample_params[k] = v
                
            elif k == "angles":
                # theta = arctan sin/cos y/x
                sample_params[k] = np.arctan2(v[:, :, 1:2], v[:, :, 0:1])
                
            elif k == "objfeats" or k == "objfeats_32":
                continue
                
            else:
                sample_params[k] = Scale.descale(
                    v, bounds[k][0], bounds[k][1]
                )
        return super().post_process(sample_params)

    @property
    def bbox_dims(self):
        return 3 + 3 + 2
    

class Scale_CosinAngle_ObjfeatsNorm(DatasetDecoratorBase):
    @staticmethod
    def scale(x, minimum, maximum):
        X = x.astype(np.float32)
        X = np.clip(X, minimum, maximum)
        X = ((X - minimum) / (maximum - minimum))
        X = 2 * X - 1
        return X

    @staticmethod
    def descale(x, minimum, maximum):
        x = (x + 1) / 2
        x = x * (maximum - minimum) + minimum
        return x

    def __getitem__(self, idx):
        bounds = self.bounds
        sample_params = self._dataset[idx]
        for k, v in sample_params.items():
            if k == "angles":
                # [cos, sin]
                sample_params[k] = np.concatenate([np.cos(v), np.sin(v)], axis=-1)

            elif k == "objfeats" or k == "objfeats_32":
                sample_params[k] = Scale.scale(
                    v, bounds[k][1], bounds[k][2]
                )
            
            elif k in bounds:
                sample_params[k] = Scale.scale(
                    v, bounds[k][0], bounds[k][1]
                )
        return sample_params

    def post_process(self, s):
        bounds = self.bounds
        sample_params = {}
        for k, v in s.items():
            if k == "room_layout" or k == "class_labels" or k == "relations" or k == "description" or k == "desc_emb":
                sample_params[k] = v
                
            elif k == "angles":
                # theta = arctan sin/cos y/x
                sample_params[k] = np.arctan2(v[:, :, 1:2], v[:, :, 0:1])
                
            elif k == "objfeats" or k == "objfeats_32":
                sample_params[k] = Scale.descale(
                    v, bounds[k][1], bounds[k][2]
                )
                
            else:
                sample_params[k] = Scale.descale(
                    v, bounds[k][0], bounds[k][1]
                )
        return super().post_process(sample_params)

    @property
    def bbox_dims(self):
        return 3 + 3 + 2


class DisturbTransOrient(DatasetDecoratorBase):
    def __getitem__(self, idx):
        sample_params = self._dataset[idx]
        for k, v in sample_params.items():
            if k == "translations":
                L, C = v.shape
                np.random.seed(idx); noise = 0.1 * np.random.randn(L, C)
                sample_params[k] = v + noise
            elif k == "angles":
                L, C = v.shape
                np.random.seed(idx*2); noise = 0.1 * np.random.randn(L, C)
                sample_params[k] = v + noise
            else:
                sample_params[k] = v
        return sample_params


class Jitter(DatasetDecoratorBase):
    def __getitem__(self, idx):
        sample_params = self._dataset[idx]
        for k, v in sample_params.items():
            if k == "room_layout" or k == "class_labels" or k == "relations" or k == "description" or k == "desc_emb" or k == "objfeats" or k == "objfeats_32":
                sample_params[k] = v
            else:
                sample_params[k] = v + np.random.normal(0, 0.01)
        return sample_params


class Permutation(DatasetDecoratorBase):
    def __init__(self, dataset, permutation_keys, permutation_axis=0):
        super().__init__(dataset)
        self._permutation_keys = permutation_keys
        self._permutation_axis = permutation_axis

    def __getitem__(self, idx):
        sample_params = self._dataset[idx]

        shapes = sample_params["class_labels"].shape
        ordering = np.random.permutation(shapes[self._permutation_axis])

        for k in self._permutation_keys:
            sample_params[k] = sample_params[k][ordering]
        return sample_params


class OrderedDataset(DatasetDecoratorBase):
    def __init__(self, dataset, ordered_keys, box_ordering=None):
        super().__init__(dataset)
        self._ordered_keys = ordered_keys
        self._box_ordering = box_ordering

    def __getitem__(self, idx):
        if self._box_ordering is None:
            return self._dataset[idx]

        if self._box_ordering != "class_frequencies":
            raise NotImplementedError()

        sample = self._dataset[idx]
        order = self._get_class_frequency_order(sample)
        for k in self._ordered_keys:
            sample[k] = sample[k][order]
        return sample

    def _get_class_frequency_order(self, sample):
        t = sample["translations"]
        c = sample["class_labels"].argmax(-1)
        class_frequencies = self.class_frequencies
        class_labels = self.class_labels
        f = np.array([
            [class_frequencies[class_labels[ci]]]
            for ci in c
        ])

        return np.lexsort(np.hstack([t, f]).T)[::-1]

####
import random
import torchtext
from num2words import num2words
from nltk.tokenize import word_tokenize
from .utils_text import compute_rel, get_article
from collections import Counter, defaultdict

def dict_bbox_to_vec(dict_box):
    '''
    input: {'min': [1,2,3], 'max': [4,5,6]}
    output: [1,2,3,4,5,6]
    '''
    return dict_box['min'] + dict_box['max']

def clean_obj_name(name):
    return name.replace('_', ' ')


class Add_Text(DatasetDecoratorBase):
    def __init__(self, dataset, eval=False, max_sentences=3, max_token_length=50): # 40
        super().__init__(dataset)
        self.eval = eval
        self.max_sentences = max_sentences
        self.glove = torchtext.vocab.GloVe(name="6B", dim=50, cache='/cluster/balrog/jtang/DiffuScene/.vector_cache') 
        self.max_token_length = max_token_length

    def __getitem__(self, idx):
        sample = self._dataset[idx]


        # Add relationship between objects
        sample = self.add_relation(sample)

        # Add description
        sample = self.add_description(sample)
        
        sample = self.add_glove_embeddings(sample)
        return sample

    def add_relation(self, sample):
        '''
            Add relations to sample['relations']
        '''
        relations = []
        num_objs = len(sample['translations'])

        for ndx in range(num_objs):
            this_box_trans = sample['translations'][ndx, :]
            this_box_sizes = sample['sizes'][ndx, :]
            this_box = {  'min': list(this_box_trans-this_box_sizes), 'max': list(this_box_trans+this_box_sizes)  }
            
            # only backward relations
            choices = [other for other in range(num_objs) if other < ndx]
            for other_ndx in choices:
                prev_box_trans = sample['translations'][other_ndx, :]
                prev_box_sizes = sample['sizes'][other_ndx, :]
                prev_box = {  'min': list(prev_box_trans-prev_box_sizes), 'max': list(prev_box_trans+prev_box_sizes) }
                box1 = dict_bbox_to_vec(this_box)
                box2 = dict_bbox_to_vec(prev_box)

                relation_str, distance = compute_rel(box1, box2)
                if relation_str is not None:
                    relation = (ndx, relation_str, other_ndx, distance)
                    relations.append(relation)
            
        sample['relations'] = relations

        return sample

    def add_description(self, sample):
        '''
            Add text descriptions to each scene
            sample['description'] = str is a sentence
            eg: 'The room contains a bed, a table and a chair. The chair is next to the window'
        '''
        sentences = []
        # clean object names once
        classes = self.class_labels
        class_index = sample['class_labels'].argmax(-1)
        obj_names = list(map(clean_obj_name, [classes[ind] for ind in class_index ] ))
        # objects that can be referred to
        refs = []
        # TODO: handle commas, use "and"
        # TODO: don't repeat, get counts and pluralize
        # describe the first 2 or 3 objects
        if self.eval:
            first_n = 3
        else:
            first_n = random.choice([2, 3])
        # first_n = len(obj_names)
        first_n_names = obj_names[:first_n] 
        first_n_counts = Counter(first_n_names)

        s = 'The room has '
        for ndx, name in enumerate(sorted(set(first_n_names), key=first_n_names.index)):
            if ndx == len(set(first_n_names)) - 1 and len(set(first_n_names)) >= 2:
                s += "and "
            if first_n_counts[name] > 1:
                s += f'{num2words(first_n_counts[name])} {name}s '
            else:
                s += f'{get_article(name)} {name} '
            if ndx == len(set(first_n_names)) - 1:
                s += ". "
            if ndx < len(set(first_n_names)) - 2:
                s += ', '
        sentences.append(s)
        refs = set(range(first_n))

        # for each object, the "position" of that object within its class
        # eg: sofa table table sofa
        #   -> 1    1    2      1
        # use this to get "first", "second"

        seen_counts = defaultdict(int)
        in_cls_pos = [0 for _ in obj_names]
        for ndx, name in enumerate(first_n_names):
            seen_counts[name] += 1
            in_cls_pos[ndx] = seen_counts[name]

        for ndx in range(1, len(obj_names)):
            # higher prob of describing the 2nd object
            prob_thresh = 0.3
                
            if self.eval:
                random_num = 1.0
            else:
                random_num = random.random() 
            if random_num > prob_thresh:
                # possible backward references for this object
                possible_relations = [r for r in sample['relations'] \
                                        if r[0] == ndx \
                                        and r[2] in refs \
                                        and r[3] < 1.5]
                if len(possible_relations) == 0:
                    continue
                # now future objects can refer to this object
                refs.add(ndx)

                # if we haven't seen this object already
                if in_cls_pos[ndx] == 0:
                    # update the number of objects of this class which have been seen
                    seen_counts[obj_names[ndx]] += 1
                    # update the in class position of this object = first, second ..
                    in_cls_pos[ndx] = seen_counts[obj_names[ndx]]

                # pick any one
                if self.eval:
                    (n1, rel, n2, dist) = possible_relations[0]
                else:
                    (n1, rel, n2, dist) = random.choice(possible_relations)
                o1 = obj_names[n1]
                o2 = obj_names[n2]

                # prepend "second", "third" for repeated objects
                if seen_counts[o1] > 1:
                    o1 = f'{num2words(in_cls_pos[n1], ordinal=True)} {o1}'
                if seen_counts[o2] > 1:
                    o2 = f'{num2words(in_cls_pos[n2], ordinal=True)} {o2}'

                # dont relate objects of the same kind
                if o1 == o2:
                    continue

                a1 = get_article(o1)

                if 'touching' in rel:
                    if ndx in (1, 2):
                        s = F'The {o1} is next to the {o2}'
                    else:
                        s = F'There is {a1} {o1} next to the {o2}'
                elif rel in ('left of', 'right of'):
                    if ndx in (1, 2):
                        s = f'The {o1} is to the {rel} the {o2}'
                    else:
                        s = f'There is {a1} {o1} to the {rel} the {o2}'
                elif rel in ('surrounding', 'inside', 'behind', 'in front of', 'on', 'above'):
                    if ndx in (1, 2):
                        s = F'The {o1} is {rel} the {o2}'
                    else:
                        s = F'There is {a1} {o1} {rel} the {o2}'
                s += ' . '
                sentences.append(s)

        # set back into the sample
        sample['description'] = sentences

        # delete sample['relations']
        del sample['relations']
        return sample

    def add_glove_embeddings(self, sample):
        sentence = ''.join(sample['description'][:self.max_sentences])
        sample['description'] = sentence
        tokens = list(word_tokenize(sentence))
        # pad to maximum length
        tokens += ['<pad>'] * (self.max_token_length - len(tokens))

        # embed words
        sample['desc_emb'] = torch.cat([self.glove[token].unsqueeze(0) for token in tokens]).numpy()

        return sample


class Autoregressive(DatasetDecoratorBase):
    def __getitem__(self, idx):
        sample_params = self._dataset[idx]

        sample_params_target = {}
        # Compute the target from the input
        for k, v in sample_params.items():
            if k == "room_layout" or k == "length":
                pass

            elif k == "relations" or k == "description" or k == "desc_emb":
                pass

            elif k == "class_labels":
                class_labels = np.copy(v)
                L, C = class_labels.shape
                # Add the end label the end of each sequence
                end_label = np.eye(C)[-1]
                sample_params_target[k+"_tr"] = np.vstack([
                    class_labels, end_label
                ])
            else:
                p = np.copy(v)
                # Set the attributes to for the end symbol
                _, C = p.shape
                sample_params_target[k+"_tr"] = np.vstack([p, np.zeros(C)])

        sample_params.update(sample_params_target)

        # Add the number of bounding boxes in the scene
        sample_params["length"] = sample_params["class_labels"].shape[0]

        return sample_params

    def collate_fn(self, samples):
        return DatasetCollection.collate_fn(samples)

    @property
    def bbox_dims(self):
        return 7


class AutoregressiveWOCM(Autoregressive):
    def __getitem__(self, idx):
        sample_params = super().__getitem__(idx)

        # Split the boxes and generate input sequences and target boxes
        L, C = sample_params["class_labels"].shape
        n_boxes = np.random.randint(0, L+1)

        for k, v in sample_params.items():
            if k == "room_layout" or k == "length":
                pass
            
            elif k == "relations" or k == "description" or k == "desc_emb":
                pass
            
            else:
                if "_tr" in k:
                    sample_params[k] = v[n_boxes]
                else:
                    sample_params[k] = v[:n_boxes]
        sample_params["length"] = n_boxes

        return sample_params

class Diffusion(DatasetDecoratorBase):
    def __getitem__(self, idx):
        sample_params = self._dataset[idx]
        max_length = self._dataset.max_length

        # Add the number of bounding boxes in the scene
        sample_params["length"] = sample_params["class_labels"].shape[0]
        
        sample_params_target = {}
        # Compute the target from the input
        for k, v in sample_params.items():
            if k == "room_layout" or k == "length":
                pass

            elif k == "relations" or k == "description" or k == "desc_emb":
                #print(k, len(v))
                pass

            elif k == "class_labels":
                class_labels = np.copy(v)
                # Delete the start label 
                new_class_labels = np.concatenate([class_labels[:, :-2], class_labels[:, -1:]], axis=-1) #hstack
                L, C = new_class_labels.shape
                # Pad the end label in the end of each sequence, and convert the class labels to -1, 1
                end_label = np.eye(C)[-1]
                sample_params_target[k] = np.vstack([
                    new_class_labels, np.tile(end_label[None, :], [max_length - L, 1])
                ]).astype(np.float32) * 2.0 - 1.0 

            else:
                p = np.copy(v)
                # Set the attributes to for the end symbol
                L, C = p.shape
                sample_params_target[k] = np.vstack([p, np.tile(np.zeros(C)[None, :], [max_length - L, 1])]).astype(np.float32)

        sample_params.update(sample_params_target)

        return sample_params
    
    def collate_fn(self, samples):
        ''' Collater that puts each data field into a tensor with outer dimension
            batch size.
        Args:
            samples: samples
        '''
    
        samples = list(filter(lambda x: x is not None, samples))
        return dataloader.default_collate(samples)

    @property
    def bbox_dims(self):
        return 7
    

def dataset_encoding_factory(
    name,
    dataset,
    augmentations=None,
    box_ordering=None
):
    # NOTE: The ordering might change after augmentations so really it should
    #       be done after the augmentations. For class frequencies it is fine
    #       though.
    if "cached" in name:
        if "objfeats" in name:
            if "lat32" in name:
                dataset_collection = OrderedDataset(
                    CachedDatasetCollection(dataset),
                    ["class_labels", "translations", "sizes", "angles", "objfeats_32"],
                    box_ordering=box_ordering
                )
                print("use lat32 as objfeats")
            else:
                dataset_collection = OrderedDataset(
                    CachedDatasetCollection(dataset),
                    ["class_labels", "translations", "sizes", "angles", "objfeats"],
                    box_ordering=box_ordering
                )
                print("use lat64 as objfeats")
        else:
            dataset_collection = OrderedDataset(
                CachedDatasetCollection(dataset),
                ["class_labels", "translations", "sizes", "angles"],
                box_ordering=box_ordering
            )
    else:
        box_ordered_dataset = BoxOrderedDataset(
            dataset,
            box_ordering
        )
        room_layout = RoomLayoutEncoder(box_ordered_dataset)
        class_labels = ClassLabelsEncoder(box_ordered_dataset)
        translations = TranslationEncoder(box_ordered_dataset)
        sizes = SizeEncoder(box_ordered_dataset)
        angles = AngleEncoder(box_ordered_dataset)
        objfeats = ObjFeatEncoder(box_ordered_dataset)
        objfeats_32 = ObjFeat32Encoder(box_ordered_dataset)

        dataset_collection = DatasetCollection(
            room_layout,
            class_labels,
            translations,
            sizes,
            angles,
            objfeats,
            objfeats_32,
        )

    if name == "basic":
        return DatasetCollection(
            class_labels,
            translations,
            sizes,
            angles, 
            objfeats,
            objfeats_32
        )

    if isinstance(augmentations, list):
        for aug_type in augmentations:
            if aug_type == "rotations":
                print("Applying rotation augmentations")
                dataset_collection = RotationAugmentation(dataset_collection)
            elif aug_type == "fixed_rotations":
                print("Applying fixed rotation augmentations")
                dataset_collection = RotationAugmentation(dataset_collection, fixed=True)
            elif aug_type == "jitter":
                print("Applying jittering augmentations")
                dataset_collection = Jitter(dataset_collection)

    if "textfix" in name:
        print("add text into input dict for evalation")
        dataset_collection = Add_Text(dataset_collection, eval=True)
    elif "text" in name:
        print("add text into input dict for training")
        dataset_collection = Add_Text(dataset_collection, eval=False)
        

    # Scale the input
    if "cosin_angle" in name or "objfeatsnorm" in name:
        print('use consin_angles instead of original angles, AND use normalized objfeats')
        dataset_collection = Scale_CosinAngle_ObjfeatsNorm(dataset_collection)
    elif "cosin_angle" in name:
        print('use consin_angles instead of original angles')
        dataset_collection = Scale_CosinAngle(dataset_collection)
    else:
        dataset_collection = Scale(dataset_collection)


    permute_keys = ["class_labels", "translations", "sizes", "angles"]
    if "objfeats" in name:
        if "lat32" in name:
            permute_keys.append("objfeats_32")
        else:
            permute_keys.append("objfeats")
    print("permute keys are:", permute_keys)
            

    # for diffusion (represent objectness as the last channel of class label)
    if "diffusion" in name:
        if "eval" in name:
            return dataset_collection
        elif "wocm_no_prm" in name:
            return Diffusion(dataset_collection)
        elif "wocm" in name:
            dataset_collection = Permutation(
                dataset_collection,
                permute_keys,
            )
            return Diffusion(dataset_collection)
        
    # for autoregressive model
    elif "autoregressive" in name:
        if "eval" in name:
            return dataset_collection
        elif "wocm_no_prm" in name:
            return AutoregressiveWOCM(dataset_collection)
        elif "wocm" in name:
            dataset_collection = Permutation(
                    dataset_collection,
                    permute_keys,
                )
            return AutoregressiveWOCM(dataset_collection)
    else:
        raise NotImplementedError()