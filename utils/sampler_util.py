import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from utils.misc import wrapped_getattr
import joblib

# A wrapper model for Classifier-free guidance **SAMPLING** only
# https://arxiv.org/abs/2207.12598
class ClassifierFreeSampleModel(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model  # model is the actual model to run

        assert self.model.cond_mask_prob > 0, 'Cannot run a guided diffusion on a model that has not been trained with no conditions'

        # pointers to inner model
        self.rot2xyz = self.model.rot2xyz
        self.translation = self.model.translation
        self.njoints = self.model.njoints
        self.nfeats = self.model.nfeats
        self.data_rep = self.model.data_rep
        self.cond_mode = self.model.cond_mode
        self.encode_text = self.model.encode_text

    def forward(self, x, timesteps, y=None):
        cond_mode = self.model.cond_mode
        assert cond_mode in ['text', 'action']
        y_uncond = deepcopy(y)
        y_uncond['uncond'] = True
        out = self.model(x, timesteps, y)
        out_uncond = self.model(x, timesteps, y_uncond)
        return out_uncond + (y['scale'].view(-1, 1, 1, 1) * (out - out_uncond))

    def __getattr__(self, name, default=None):
        # this method is reached only if name is not in self.__dict__.
        return wrapped_getattr(self, name, default=None)


class AutoRegressiveSampler():
    def __init__(self, args, sample_fn, required_frames=196):
        self.sample_fn = sample_fn
        self.args = args
        self.required_frames = required_frames
    
    def sample(self, model, shape, **kargs):
        bs = shape[0]
        n_iterations = (self.required_frames // self.args.pred_len) + int(self.required_frames % self.args.pred_len > 0)
        samples_buf = []
        cur_prefix = deepcopy(kargs['model_kwargs']['y']['prefix'])  # init with data
        dynamic_text_mode = type(kargs['model_kwargs']['y']['text'][0]) == list  # Text changes on the fly - prompt per prediction is provided as a list (instead of a single prompt)
        if self.args.autoregressive_include_prefix:
            samples_buf.append(cur_prefix)
        autoregressive_shape = list(deepcopy(shape))
        autoregressive_shape[-1] = self.args.pred_len
        
        # Autoregressive sampling
        for i in range(n_iterations):
            
            # Build the current kargs
            cur_kargs = deepcopy(kargs)
            cur_kargs['model_kwargs']['y']['prefix'] = cur_prefix
            if dynamic_text_mode:
                cur_kargs['model_kwargs']['y']['text'] = [s[i] for s in kargs['model_kwargs']['y']['text']]
                if model.text_encoder_type == 'bert':
                    cur_kargs['model_kwargs']['y']['text_embed'] = (cur_kargs['model_kwargs']['y']['text_embed'][0][:, :, i], cur_kargs['model_kwargs']['y']['text_embed'][1][:, i])
                else:
                    raise NotImplementedError('DiP model only supports BERT text encoder at the moment. If you implement this, please send a PR!')
            
            # Sample the next prediction
            sample = self.sample_fn(model, autoregressive_shape, **cur_kargs)

            # Buffer the sample
            samples_buf.append(sample.clone()[..., -self.args.pred_len:])

            # Update the prefix
            cur_prefix = sample.clone()[..., -self.args.context_len:]

        full_batch = torch.cat(samples_buf, dim=-1)[..., :self.required_frames]  # 200 -> 196
        return full_batch

class PredictionAutoRegressiveSampler():
    def __init__(self, args, sample_fn, required_frames, history_len):
        self.sample_fn = sample_fn
        self.args = args
        self.required_frames = required_frames
        self.history_frames = history_len
    
    def sample(self, model, shape, **kargs):
        # shape is the shape of the single prediction window: [bs, njoints, nfeats, pred_len]
        bs = shape[0]
        
        # The step size is fixed to exactly the history window size. 
        # This ensures the input to the next step is always the first `history_frames` of the new generation window.
        step_size = self.history_frames
        overlap_len = self.args.pred_len - step_size

        # If overlap is 0 or invalidly computed backward, fallback to standard step
        if overlap_len <= 0 or step_size <= 0:
            overlap_len = 0
            step_size = self.args.pred_len
            
        # In variable length prediction, lengths tells us exactly how long each sample in the batch should be generated
        if 'lengths' in kargs['model_kwargs']['y']:
            lengths = kargs['model_kwargs']['y']['lengths']
            max_frames = int(lengths.max().item())
        else:
            max_frames = self.required_frames
            
        # Calculate how many iterations we need
        n_iterations = (max_frames // step_size) + int(max_frames % step_size > 0)
        
        device = next(model.parameters()).device
        
        # We will accumulate the full generated sequence into one large tensor, padding with zeros initially
        full_sequence_shape = list(shape)
        full_sequence_shape[-1] = n_iterations * step_size + overlap_len
        if hasattr(model, 'dtype'):
            seq_dtype = model.dtype
        else:
            seq_dtype = next(model.parameters()).dtype
        full_sequence = torch.zeros(full_sequence_shape, dtype=seq_dtype, device=device)
        
        # The blending weight dictates the ratio of (previous_prediction * weight + next_prediction * (1 - weight))
        if overlap_len > 0:
            import math
            steps = torch.linspace(0.0, 1.0, overlap_len, device=device)
            blend_w = (1.0 + torch.cos(math.pi * steps)) / 2.0
            blend_w = blend_w.view(1, 1, 1, overlap_len).to(seq_dtype)
        
        cur_history = deepcopy(kargs['model_kwargs']['y']['history']).to(device)
        dynamic_text_mode = type(kargs['model_kwargs']['y'].get('text', None)) == list and isinstance(kargs['model_kwargs']['y']['text'][0], list)

        for i in range(n_iterations):
            cur_kargs = deepcopy(kargs)
            cur_kargs['model_kwargs']['y']['history'] = cur_history
            if 'mask' in cur_kargs['model_kwargs']['y']:
                # The generation window is fully valid by definition, so we provide an unmasked window of pred_len
                cur_kargs['model_kwargs']['y']['mask'] = torch.ones((bs, 1, 1, self.args.pred_len), dtype=torch.bool, device=cur_history.device)

            # Handle dynamic text prompts (if any)
            if dynamic_text_mode:
                cur_kargs['model_kwargs']['y']['text'] = [s[i] for s in kargs['model_kwargs']['y']['text']]
                if model.text_encoder_type == 'bert':
                    cur_kargs['model_kwargs']['y']['text_embed'] = (
                        cur_kargs['model_kwargs']['y']['text_embed'][0][:, :, i],
                        cur_kargs['model_kwargs']['y']['text_embed'][1][:, i]
                    )
                else:
                    raise NotImplementedError('Dynamic text only supported with BERT.')

            # Ensure all tensor kwargs in 'y' are located on model device before querying the sampler
            cur_kargs['model_kwargs']['y'] = {k: v.to(device) if torch.is_tensor(v) else v for k, v in cur_kargs['model_kwargs']['y'].items()}

            # Generate the future (returns [bs, njoints, nfeats, pred_len])
            sample_t = self.sample_fn(model, shape, **cur_kargs)
            
            start_idx = i * step_size
            end_idx = start_idx + self.args.pred_len
            
            if i == 0 or overlap_len == 0:
                # First iteration: just plonk it down completely
                full_sequence[..., start_idx:end_idx] = sample_t
            else:
                # Blend the overlapping chunk
                previous_tail = full_sequence[..., start_idx : start_idx + overlap_len]
                sample_t_head = sample_t[..., :overlap_len]
                
                blended_overlap = previous_tail * blend_w + sample_t_head * (1.0 - blend_w)
                
                # Overwrite the overlap region
                full_sequence[..., start_idx : start_idx + overlap_len] = blended_overlap
                # Write the new un-overlapped tail
                full_sequence[..., start_idx + overlap_len : end_idx] = sample_t[..., overlap_len:]

            # Update history: we need the LAST `history_frames` from the sequence we've confirmed so far
            # The confirmed sequence (safe to use as history) goes up to (start_idx + step_size) before the next iteration
            next_start_idx = (i + 1) * step_size
            
            # Temporary concatenation array holding true GT history and the generated stable sequence
            original_history_t = kargs['model_kwargs']['y']['history'].to(full_sequence.device)
            history_pool = torch.cat((original_history_t, full_sequence[..., :next_start_idx]), dim=-1)
            cur_history = history_pool[..., -self.history_frames:].clone()

        # Extract only the required frames and combine with original prefix if requested
        generated_frames = full_sequence[..., :max_frames]

        if self.args.autoregressive_include_prefix:
            original_history_t = kargs['model_kwargs']['y']['history'].to(full_sequence.device)
            generated_frames = torch.cat((original_history_t, generated_frames), dim=-1)

        return generated_frames