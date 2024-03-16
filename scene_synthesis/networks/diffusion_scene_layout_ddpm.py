from curses import noecho
from doctest import debug_script
import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn.utils import clip_grad_norm_

from .diffusion_ddpm import DiffusionPoint
from .denoise_net import Unet1D
from ..stats_logger import StatsLogger
from transformers import BertTokenizer, BertModel
import clip

class DiffusionSceneLayout_DDPM(Module):

    def __init__(self, n_classes, feature_extractor, config):
        super().__init__()

        # TODO: Add the projection dimensions for the room features in the
        # config!!!

        # if use room_mask_condition: if yes, define the feature extractor for the room mask
        self.room_mask_condition = config.get("room_mask_condition", True)
        self.text_condition = config.get("text_condition", False)
        self.text_glove_embedding = config.get("text_glove_embedding", False)
        self.text_clip_embedding = config.get("text_clip_embedding", False)
        if self.room_mask_condition:
            self.feature_extractor = feature_extractor
            self.fc_room_f = nn.Linear(
                self.feature_extractor.feature_size, config["latent_dim"]
            )
            print('use room mask as condition')
        elif self.text_condition:
            text_embed_dim = config.get("text_embed_dim", 512)

            if self.text_glove_embedding:
                self.fc_text_f = nn.Linear(50, text_embed_dim)
                print('use text as condition, and pretrained glove embedding')
            elif self.text_clip_embedding:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)

                for p in self.clip_model.parameters():
                    p.requires_grad = False
                print('use text as condition, and pretrained clip embedding')
            else:
                self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
                self.bertmodel = BertModel.from_pretrained("bert-base-cased")

                for p in self.bertmodel.parameters():
                    p.requires_grad = False
                self.fc_text_f = nn.Linear(768, text_embed_dim)
                print('use text as condition, and pretrained bert model')

        else:
            print('NOT use room and text as condition')

        # define the denoising network
        if config["net_type"] == "unet1d":
            denoise_net = Unet1D(**config["net_kwargs"])
        else:
            raise NotImplementedError()

        # define the diffusion type
        self.diffusion = DiffusionPoint(
            denoise_net = denoise_net,
            config = config,
            **config["diffusion_kwargs"]
        )
        self.n_classes = n_classes
        self.config = config
        
        # read object property dimension
        self.objectness_dim = config.get("objectness_dim", 1)
        self.class_dim = config.get("class_dim", 21)
        self.translation_dim = config.get("translation_dim", 3)
        self.size_dim = config.get("size_dim", 3)
        self.angle_dim = config.get("angle_dim", 1)
        self.bbox_dim = self.translation_dim + self.size_dim + self.angle_dim
        self.objfeat_dim = config.get("objfeat_dim", 0)

        # define class and instance embeddings
        self.learnable_embedding = config.get("learnable_embedding", False)
        self.instance_condition = config.get("instance_condition", False)
        self.sample_num_points = config.get("sample_num_points", 12)
        self.instance_emb_dim = config.get("instance_emb_dim", 64)
        
        if self.learnable_embedding:
            if self.instance_condition:
                self.register_parameter(
                    "positional_embedding",
                    nn.Parameter(torch.randn(self.sample_num_points, self.instance_emb_dim))
                )
            else:
                self.instance_emb_dim = 0
    
        else:
            if self.instance_condition:
                self.fc_instance_condition = nn.Sequential(
                    nn.Linear(self.sample_num_points, self.instance_emb_dim, bias=False),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Linear(self.instance_emb_dim, self.instance_emb_dim, bias=False),
                )
            else:
                self.instance_emb_dim = 0

        # defind other kinds of condition: partial objects or scene arrangement (size, class, objectness, and objfeats)
        self.room_partial_condition = config.get("room_partial_condition", False)
        self.partial_num_points = config.get("partial_num_points", 0)
        self.partial_emb_dim = config.get("partial_emb_dim", 64)
        if self.room_partial_condition:
            self.fc_partial_condition = nn.Sequential(
                    nn.Linear(self.bbox_dim+self.class_dim+self.objectness_dim+self.objfeat_dim, self.partial_emb_dim, bias=False),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Linear(self.partial_emb_dim, self.partial_emb_dim, bias=False),
                )
        else:
            self.partial_emb_dim = 0

        self.room_arrange_condition = config.get("room_arrange_condition", False)
        self.arrange_emb_dim = config.get("arrange_emb_dim", 64)
        if self.room_arrange_condition:
            self.fc_arrange_condition = nn.Sequential(
                    nn.Linear(self.size_dim+self.class_dim+self.objectness_dim+self.objfeat_dim, self.arrange_emb_dim, bias=False),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Linear(self.arrange_emb_dim, self.arrange_emb_dim, bias=False),
                )
        else:
            self.arrange_emb_dim = 0

    def get_loss(self, sample_params):
        # Unpack the sample_params
        if self.objectness_dim >0:
            objectness   = sample_params["objectness"]
        class_labels = sample_params["class_labels"]
        translations = sample_params["translations"]
        sizes = sample_params["sizes"]
        angles = sample_params["angles"]
        if self.objfeat_dim >0:
            if self.objfeat_dim == 32:
                objfeats = sample_params["objfeats_32"]
            else:
                objfeats = sample_params["objfeats"]
        room_layout = sample_params["room_layout"]
        batch_size, num_points, _ = class_labels.shape

        # get desired diffusion target
        if self.config["point_dim"] == self.bbox_dim+self.class_dim+self.objectness_dim+self.objfeat_dim:
            if self.objectness_dim>0:
                room_layout_target = torch.cat([translations, sizes, angles, class_labels, objectness], dim=-1).contiguous()   
            else:
                room_layout_target = torch.cat([translations, sizes, angles, class_labels], dim=-1).contiguous() 
            if self.objfeat_dim > 0:
                room_layout_target = torch.cat([room_layout_target, objfeats], dim=-1).contiguous() 

        elif self.config["point_dim"] == self.bbox_dim:
            room_layout_target = torch.cat([translations, sizes, angles], dim=-1).contiguous()  
    
        else:
            raise NotImplementedError

        # get the latent feature of room_mask
        if self.room_mask_condition:
            room_layout_f = self.fc_room_f(self.feature_extractor(room_layout)) #(B, F)
            
        else:
            room_layout_f = None

        device = class_labels.device

        # process instance & class condition f
        if self.instance_condition:
            if self.learnable_embedding:
                instance_indices = torch.arange(self.sample_num_points).long().to(device)[None, :].repeat(batch_size, 1)
                instan_condition_f = self.positional_embedding[instance_indices, :]
            else:
                instance_label = torch.eye(self.sample_num_points).float().to(device)[None, ...].repeat(batch_size, 1, 1)
                instan_condition_f = self.fc_instance_condition(instance_label) 
        else:
            instan_condition_f = None

        # concat instance and class condition
        # concat room_layout_f and instan_class_f
        if room_layout_f is not None and instan_condition_f is not None:
            condition = torch.cat([room_layout_f[:, None, :].repeat(1, num_points, 1), instan_condition_f], dim=-1).contiguous()
        elif room_layout_f is not None:
            condition = room_layout_f[:, None, :].repeat(1, num_points, 1)
        elif instan_condition_f is not None:
            condition = instan_condition_f
        else:
            condition = None

        # concat room_partial  condition
        if self.room_partial_condition:
            partial_valid   = torch.ones((batch_size, self.partial_num_points, 1)).float().to(device)
            partial_invalid = torch.zeros((batch_size, num_points - self.partial_num_points, 1)).float().to(device)
            partial_mask    = torch.cat([ partial_valid, partial_invalid ], dim=1).contiguous()
            partial_input   = room_layout_target * partial_mask
            partial_condition_f = self.fc_partial_condition(partial_input)
            condition = torch.cat([condition, partial_condition_f], dim=-1).contiguous()

        # concat room_arrange condition
        if self.room_arrange_condition:
            arrange_input  = torch.cat([ room_layout_target[:, :, self.translation_dim:self.translation_dim+self.size_dim], room_layout_target[:, :, self.bbox_dim:] ], dim=-1).contiguous()
            arrange_condition_f = self.fc_arrange_condition(arrange_input)
            condition = torch.cat([condition, arrange_condition_f], dim=-1).contiguous()
            room_layout_target  = torch.cat([ room_layout_target[:, :, 0:self.translation_dim], room_layout_target[:, :, self.translation_dim+self.size_dim:self.bbox_dim] ], dim=-1).contiguous()

        # use text embed for cross attention
        if self.text_condition:
            if self.text_glove_embedding:
                condition_cross = self.fc_text_f( sample_params["desc_emb"] ) 
            elif self.text_clip_embedding:
                tokenized = clip.tokenize(sample_params["description"]).to(device)
                condition_cross = self.clip_model.encode_text(tokenized)
            else:
                tokenized = self.tokenizer(sample_params["description"], return_tensors='pt',padding=True).to(device)
                text_f = self.bertmodel(**tokenized).last_hidden_state
                condition_cross = self.fc_text_f( text_f )
        else:
            condition_cross = None

        # denoise loss function
        loss, loss_dict = self.diffusion.get_loss_iter(room_layout_target, condition=condition, condition_cross=condition_cross)

        return loss, loss_dict

    def sample(self, room_mask, num_points, point_dim, batch_size=1, text=None, 
               partial_boxes=None, input_boxes=None, ret_traj=False, ddim=False, clip_denoised=False, freq=40, batch_seeds=None, 
                ):
        device = room_mask.device
        noise = torch.randn((batch_size, num_points, point_dim))#, device=room_mask.device)

        # get the latent feature of room_mask
        if self.room_mask_condition:
            room_layout_f = self.fc_room_f(self.feature_extractor(room_mask)) #(B, F)
            
        else:
            room_layout_f = None

        # process instance & class condition f
        if self.instance_condition:
            if self.learnable_embedding:
                instance_indices = torch.arange(self.sample_num_points).long().to(device)[None, :].repeat(room_mask.size(0), 1)
                instan_condition_f = self.positional_embedding[instance_indices, :]
            else:
                instance_label = torch.eye(self.sample_num_points).float().to(device)[None, ...].repeat(room_mask.size(0), 1, 1)
                instan_condition_f = self.fc_instance_condition(instance_label) 
        else:
            instan_condition_f = None


        # concat instance and class condition   
        # concat room_layout_f and instan_class_f
        if room_layout_f is not None and instan_condition_f is not None:
            condition = torch.cat([room_layout_f[:, None, :].repeat(1, num_points, 1), instan_condition_f], dim=-1).contiguous()
        elif room_layout_f is not None:
            condition = room_layout_f[:, None, :].repeat(1, num_points, 1)
        elif instan_condition_f is not None:
            condition = instan_condition_f
        else:
            condition = None

        # concat room_partial condition, use partial boxes as input for scene completion
        if self.room_partial_condition:
            zeros_boxes = torch.zeros((batch_size, num_points-partial_boxes.shape[1], partial_boxes.shape[2])).float().to(device)
            partial_input  =  torch.cat([partial_boxes, zeros_boxes], dim=1).contiguous()
            partial_condition_f = self.fc_partial_condition(partial_input)
            condition = torch.cat([condition, partial_condition_f], dim=-1).contiguous()

        # concat  room_arrange condition, use input boxes as input for scene completion
        if self.room_arrange_condition:
            arrange_input  = torch.cat([ input_boxes[:, :, self.translation_dim:self.translation_dim+self.size_dim], input_boxes[:, :, self.bbox_dim:] ], dim=-1).contiguous()
            arrange_condition_f = self.fc_arrange_condition(arrange_input)
            condition = torch.cat([condition, arrange_condition_f], dim=-1).contiguous()


        if self.text_condition:
            if self.text_glove_embedding:
                condition_cross = self.fc_text_f(text) #sample_params["desc_emb"]
            elif self.text_clip_embedding:
                tokenized = clip.tokenize(text).to(device)
                condition_cross = self.clip_model.encode_text(tokenized)
            else:
                tokenized = self.tokenizer(text, return_tensors='pt',padding=True).to(device)
                #print('tokenized:', tokenized.shape)
                text_f = self.bertmodel(**tokenized).last_hidden_state
                print('after bert:', text_f.shape)
                condition_cross = self.fc_text_f( text_f )
        else:
            condition_cross = None
            

        if input_boxes is not None:
            print('scene arrangement sampling')
            samples = self.diffusion.arrange_samples(noise.shape, room_mask.device, condition=condition, condition_cross=condition_cross, clip_denoised=clip_denoised, input_boxes=input_boxes)

        elif partial_boxes is not None:
            print('scene completion sampling')
            samples = self.diffusion.complete_samples(noise.shape, room_mask.device, condition=condition, condition_cross=condition_cross, clip_denoised=clip_denoised, partial_boxes=partial_boxes)

        else:
            print('unconditional / conditional generation sampling')
            # reverse sampling
            if ret_traj:
                samples = self.diffusion.gen_sample_traj(noise.shape, room_mask.device, freq=freq, condition=condition, condition_cross=condition_cross, clip_denoised=clip_denoised)
            else:
                samples = self.diffusion.gen_samples(noise.shape, room_mask.device, condition=condition, condition_cross=condition_cross, clip_denoised=clip_denoised)
            
        return samples

    @torch.no_grad()
    def generate_layout(self, room_mask, num_points, point_dim, batch_size=1, text=None, ret_traj=False, ddim=False, clip_denoised=False, batch_seeds=None, device="cpu", keep_empty=False):
        
        samples = self.sample(room_mask, num_points, point_dim, batch_size, text=text, ret_traj=ret_traj, ddim=ddim, clip_denoised=clip_denoised, batch_seeds=batch_seeds)
        
        return self.delete_empty_from_network_samples(samples, device=device, keep_empty=keep_empty)

    @torch.no_grad()
    def generate_layout_progressive(self, room_mask, num_points, point_dim, batch_size=1, text=None, ret_traj=False, ddim=False, clip_denoised=False, batch_seeds=None, device="cpu", keep_empty=False, num_step=100):
        
        # output dictionary of sample trajectory & sample some key steps
        samples_traj = self.sample(room_mask, num_points, point_dim, batch_size, text=text, ret_traj=ret_traj, ddim=ddim, clip_denoised=clip_denoised, batch_seeds=batch_seeds, freq=num_step)
        boxes_traj = {}

        # delete the initial noisy
        samples_traj = samples_traj[1:]

        for i in range(len(samples_traj)):
            samples = samples_traj[i]
            k_time = num_step * i
            boxes_traj[k_time] = self.delete_empty_from_network_samples(samples, device=device, keep_empty=keep_empty)
        return boxes_traj
    
    @torch.no_grad()
    def complete_scene(self, room_mask, num_points, point_dim, partial_boxes, batch_size=1, ret_traj=False, ddim=False, clip_denoised=False, batch_seeds=None, device="cpu", keep_empty=False):
        
        samples = self.sample(room_mask, num_points, point_dim, batch_size, partial_boxes=partial_boxes, ret_traj=ret_traj, ddim=ddim, clip_denoised=clip_denoised, batch_seeds=batch_seeds)

        return self.delete_empty_from_network_samples(samples, device=device, keep_empty=keep_empty)
    
    @torch.no_grad()
    def arrange_scene(self, room_mask, num_points, point_dim, input_boxes, batch_size=1, ret_traj=False, ddim=False, clip_denoised=False, batch_seeds=None, device="cpu", keep_empty=False):
        
        samples = self.sample(room_mask, num_points, point_dim, batch_size, input_boxes=input_boxes, ret_traj=ret_traj, ddim=ddim, clip_denoised=clip_denoised, batch_seeds=batch_seeds)

        return self.delete_empty_from_network_samples(samples, device=device, keep_empty=keep_empty)
    
    

    @torch.no_grad()
    def delete_empty_from_network_samples(self, samples, device="cpu", keep_empty=False):
        
        samples_dict = {
            "translations": samples[:, :, 0:self.translation_dim].contiguous(),
            "sizes": samples[:, :,  self.translation_dim:self.translation_dim+self.size_dim].contiguous(),
            "angles": samples[:, :, self.translation_dim+self.size_dim:self.bbox_dim].contiguous(),
            "class_labels": nn.functional.one_hot( torch.argmax(samples[:, :, self.bbox_dim:self.bbox_dim+self.class_dim-1].contiguous(), dim=-1), \
                            num_classes=self.n_classes-2),
            "objectness": samples[:, :, self.bbox_dim+self.class_dim-1:self.bbox_dim+self.class_dim]>=0,
        }
        if self.objfeat_dim > 0:
            samples_dict["objfeats"] = samples[:, :, self.bbox_dim+self.class_dim:self.bbox_dim+self.class_dim+self.objfeat_dim]

        #initilization
        boxes = {
            "objectness": torch.zeros(1, 0, 1, device=device),
            "class_labels": torch.zeros(1, 0, self.n_classes-2, device=device),
            "translations": torch.zeros(1, 0, self.translation_dim, device=device),
            "sizes": torch.zeros(1, 0, self.size_dim, device=device),
            "angles": torch.zeros(1, 0, self.angle_dim, device=device)
        }
        if self.objfeat_dim > 0:
            boxes["objfeats"] =  torch.zeros(1, 0, self.objfeat_dim, device=device)
    
        max_boxes = samples.shape[1]
        for i in range(max_boxes):
            # Check if we have the end symbol 
            if not keep_empty and samples_dict['objectness'][0, i, -1] > 0:
                continue
            else:
                for k in samples_dict.keys():
                    if k == "class_labels":
                        # we output raw probability maps for visualization
                        boxes[k] = torch.cat([ boxes[k], samples[:, i:i+1, self.bbox_dim:self.bbox_dim+self.class_dim-1].to(device) ], dim=1)
                        boxes["objectness"] = torch.cat([ boxes["objectness"], samples[:, i:i+1, self.bbox_dim+self.class_dim-1:self.bbox_dim+self.class_dim].to(device) ], dim=1)
                    else:
                        boxes[k] = torch.cat([ boxes[k], samples_dict[k][:, i:i+1, :].to(device) ], dim=1)

        if self.objfeat_dim > 0:
            return {
            "class_labels": boxes["class_labels"].to("cpu"),
            #"objectness": boxes["objectness"].to("cpu"),
            "translations": boxes["translations"].to("cpu"),
            "sizes": boxes["sizes"].to("cpu"),
            "angles": boxes["angles"].to("cpu"),
            "objfeats": boxes["objfeats"].to("cpu"),
        }
        else:
            return {
                "class_labels": boxes["class_labels"].to("cpu"),
                #"objectness": boxes["objectness"].to("cpu"),
                "translations": boxes["translations"].to("cpu"),
                "sizes": boxes["sizes"].to("cpu"),
                "angles": boxes["angles"].to("cpu")
            }


    @torch.no_grad()
    def delete_empty_boxes(self, samples_dict, device="cpu", keep_empty=False):

        #initilization
        boxes = {
            "objectness": torch.zeros(1, 0, 1, device=device),
            "class_labels": torch.zeros(1, 0, self.n_classes-2, device=device),
            "translations": torch.zeros(1, 0, self.translation_dim, device=device),
            "sizes": torch.zeros(1, 0, self.size_dim, device=device),
            "angles": torch.zeros(1, 0, self.angle_dim, device=device)
        }
        if self.objfeat_dim > 0:
            boxes["objfeats"] =  torch.zeros(1, 0, self.objfeat_dim, device=device)
    
        max_boxes = samples_dict["class_labels"].shape[1]
        for i in range(max_boxes):
            # Check if we have the end symbol 
            if not keep_empty and samples_dict['class_labels'][0, i, -1] > 0:
                continue
            else:
                for k in samples_dict.keys():
                    if k == "class_labels":
                        # we output raw probability maps for visualization
                        boxes[k] = torch.cat([ boxes[k], samples_dict[k][:, i:i+1, :self.class_dim-1].to(device) ], dim=1)
                        boxes["objectness"] = torch.cat([ boxes["objectness"], samples_dict[k][:, i:i+1, -1:].to(device) ], dim=1)
                    else:
                        boxes[k] = torch.cat([ boxes[k], samples_dict[k][:, i:i+1, :].to(device) ], dim=1)

        
        if self.objfeat_dim > 0:
                return {
                "class_labels": boxes["class_labels"].to("cpu"),
                #"objectness": boxes["objectness"].to("cpu"),
                "translations": boxes["translations"].to("cpu"),
                "sizes": boxes["sizes"].to("cpu"),
                "angles": boxes["angles"].to("cpu"),
                "objfeats": boxes["objfeats"].to("cpu"),
            }
        else:
            return {
                "class_labels": boxes["class_labels"].to("cpu"),
                #"objectness": boxes["objectness"].to("cpu"),
                "translations": boxes["translations"].to("cpu"),
                "sizes": boxes["sizes"].to("cpu"),
                "angles": boxes["angles"].to("cpu"),
            }

def train_on_batch(model, optimizer, sample_params, config):
    # Make sure that everything has the correct size
    optimizer.zero_grad()
    # Compute the loss
    loss, loss_dict = model.get_loss(sample_params)
    for k, v in loss_dict.items():
        StatsLogger.instance()[k].value = v.item()
    # Do the backpropagation
    loss.backward()
    # Compuite model norm
    grad_norm = clip_grad_norm_(model.parameters(), config["training"]["max_grad_norm"])
    StatsLogger.instance()["gradnorm"].value = grad_norm.item()
    # log learning rate
    StatsLogger.instance()["lr"].value = optimizer.param_groups[0]['lr']
    # Do the update
    optimizer.step()

    return loss.item()


@torch.no_grad()
def validate_on_batch(model, sample_params, config):
    # Compute the loss
    loss, loss_dict = model.get_loss(sample_params)
    for k, v in loss_dict.items():
        StatsLogger.instance()[k].value = v.item()
    return loss.item()
