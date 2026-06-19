#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 00:06:59 2026

@author: asandhir
"""

import numpy as np

import torch
import torch.optim as optim
from torch.nn import CTCLoss
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from IPython.display import Audio, display
from tqdm import trange

class Backmasking:
    def __init__(self, audio_path = 'inputs/sample.ogg', 
                 binary_search_steps = 5, initial_constant = .001, 
                 lr = 0.01, max_iterations = 500):
        
        # load model and its utilities
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' 
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(self.device)
        self.required_sampling_rate = self.processor.feature_extractor.sampling_rate
        self.ctc_loss_fn = CTCLoss(blank=self.processor.tokenizer.pad_token_id, zero_infinity=True)

        # process audio
        self.audio_path = audio_path
        self.forward_waveform = self.preprocess_audio(audio_path)
        self.forward_transcription = self.transcribe(self.forward_waveform)
        self.reverse_waveform = self.reverse_audio(self.forward_waveform)
        self.reverse_transcription = self.transcribe(self.reverse_waveform)
        
        # perform Carlini-Wagner attack
        self.binary_search_steps = binary_search_steps
        self.initial_constant = initial_constant
        self.lr = lr
        self.max_iterations = max_iterations
        self.target = self.select_target()
        self.adversarial_waveform = self.CW_outter(self.reverse_waveform, self.target)
        self.converged = (self.adversarial_waveform != self.reverse_waveform).any().item()
        
    def preprocess_audio(self, audio_path):
        """
        Loads an audio file, and returns a single channel signal sampled at 16kHz

        Parameters
        ----------
        audio_path : string
            file path pointing to an audio file, .ogg and .wav are supported.

        Returns
        -------
        waveform : torch.Tensor
            DESCRIPTION.

        """
        waveform, sample_rate = torchaudio.load(audio_path)
        # wav2vec2 expects 16kHz audio
        if sample_rate != self.required_sampling_rate:
            waveform = torchaudio.transforms.Resample(sample_rate, self.required_sampling_rate)(waveform)
        
        # mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        return waveform
    
    def play_audio(self, waveform):
        display(Audio(waveform, rate=self.required_sampling_rate))
    
    def transcribe(self, waveform):
        """
        Infers upon the Wav2Vec2 model and returns the transcript of a given 
        audio waveform

        Parameters
        ----------
        waveform : torch.Tensor
            DESCRIPTION.

        Returns
        -------
        transcription : string
            DESCRIPTION.

        """
        with torch.no_grad():
            logits = self.model(waveform).logits
        
        pred_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.decode(pred_ids[0])
        
        return transcription
    
    def reverse_audio(self, waveform):
        """
        Reverses the order of a given audio waveform

        Parameters
        ----------
        waveform : torch.Tensor
            DESCRIPTION.

        Returns
        -------
        reverse : torch.Tensor
            DESCRIPTION.

        """
        reverse = torch.flip(waveform, dims = [1])
    
        return reverse
    
    def select_target(self):
        """
        Randomly selects a quote from targets.txt to serve as the target 
        transcription for the upcomming Carlini-Wagner attack

        Returns
        -------
        target : string
            DESCRIPTION.

        """
        with open('inputs/targets.txt', mode='r', encoding='utf-8') as file:
            targets = file.read()
            
        targets = targets.split('\n')
        
        random_number_generator = np.random.default_rng()
        target = random_number_generator.choice(targets, size=1)[0].upper()
            
        return target
    
    def CW_outter(self, waveform, target):
        
        yet_to_converge = True
        c = self.initial_constant
        lowerbound_c = 0
        upperbound_c = 1e9
        best_adverserial_audio = waveform
        for step in range(self.binary_search_steps):
            adverserial_audio, converged = self.CW_inner(waveform, target, c)
            if (converged == True):
                yet_to_converge = False
                upperbound_c = c 
                c = (lowerbound_c + upperbound_c)/2
                best_adverserial_audio = adverserial_audio
            elif (converged == False) and (yet_to_converge == False):
                lowerbound_c = c 
                c = (lowerbound_c + upperbound_c)/2
            elif (converged == False) and (yet_to_converge == True):
                lowerbound_c = c 
                c *= 10
        
        return best_adverserial_audio
                
    def CW_inner(self, waveform, target, c):
        """
        original_waveform: 1D torch.Tensor of audio at 16000Hz
        target: string (e.g., "TAKE ME TO YOUR LEADER")
        """        
        # 1. Prepare target tokens
        target_tokens = self.processor(text=target, return_tensors="pt").input_ids[0].to(self.device)
        target_length = torch.tensor([len(target_tokens)], dtype=torch.long).to(self.device)
        
        # 2. Normalize the original audio (matching Wav2Vec2's typical input style)
        # Note: If your pipeline handles normalization inside the forward pass, adjust accordingly
        x = waveform.clone().detach().to(self.device)# Shape: [1, seq_len]
        w = torch.atanh((1 - 1e-6) * (waveform))
    
        w_perturbations = torch.zeros_like(w, requires_grad=True, device=self.device)
        optimizer = optim.Adam([w_perturbations], lr=self.lr)
        converged = False

        progress_bar = trange(self.max_iterations)
        for iteration in progress_bar:
            optimizer.zero_grad()
            
            # The adversarial audio sample
            adversarial_audio = torch.tanh(w + w_perturbations)
            
            # Forward pass through Wav2Vec2 model to get raw logits
            # Depending on your architecture, you may need to compute features first
            logits = self.model(adversarial_audio).logits  # Shape: [1, frame_steps, vocab_size]
            
            # Logits formatting for PyTorch CTCLoss: [frame_steps, batch_size, vocab_size]
            logits_log_probs = logits.log_softmax(2).transpose(0, 1)
            input_length = torch.tensor([logits_log_probs.size(0)], dtype=torch.long).to(self.device)
            
            # Calculate ASR Target Loss (CTC Loss)
            loss_ctc = self.ctc_loss_fn(logits_log_probs, target_tokens, input_length, target_length)
            
            # Calculate Distortion Loss (L2 Norm of the perturbation)
            loss_distortion = torch.mean(w_perturbations ** 2)
            
            # Total Loss formulation
            total_loss = loss_distortion + (c * loss_ctc)
            
            # Backward pass to find gradients w.r.t 'delta'
            total_loss.backward()
            optimizer.step()
            
            predicted_ids = torch.argmax(logits, dim=-1)
            current_transcription = self.processor.batch_decode(predicted_ids)[0]
            #status_bar.set_description_str(f"Transcript: {current_transcription}")
            
            progress_bar.set_postfix(
                c=f"{c}",
                Loss=f"{total_loss.item():.4f}",
                L2=f"{loss_distortion.item():.4f}"
            )
                       
            # Dynamic scaling (if target is reached, focus more on minimizing distortion)
            if current_transcription == target:
                converged = True
                best_adverserial_example = torch.tanh(w + w_perturbations).detach().cpu().squeeze(0)
                print("--> Success! Target achieved. Refining imperceptibility...")
                break
            
            best_adverserial_example = torch.tanh(w + w_perturbations).detach().cpu().squeeze(0)
            
        return best_adverserial_example, converged
    
if __name__ == "__main__":
    backmask = Backmasking(binary_search_steps = 3, initial_constant = .000125,
                           max_iterations=200)
    print(backmask.forward_transcription)
    print(backmask.reverse_transcription)
    print(backmask.target)
    
    backmask.play_audio(backmask.forward_waveform)
    backmask.play_audio(backmask.reverse_waveform)
    if backmask.converged:
        backmask.play_audio(backmask.adversarial_waveform)