#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 00:06:59 2026

@author: asandhir
"""

import io
from pathlib import Path
import numpy as np

import torch
import torch.optim as optim
from torch.nn import CTCLoss
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from IPython.display import Audio, display, HTML
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import trange

class Backmasking:
    def __init__(self, audio_path = 'inputs/sample.ogg', 
                 outputs_dir = 'animations', binary_search_steps = 5, 
                 initial_constant = .001, lr = 0.01, max_iterations = 500):
        
        # load model and its utilities
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' 
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(self.device)
        self.required_sampling_rate = self.processor.feature_extractor.sampling_rate
        self.ctc_loss_fn = CTCLoss(blank=self.processor.tokenizer.pad_token_id, zero_infinity=True)
        self.spectrogram_transform = torchaudio.transforms.Spectrogram(
            n_fft=1024,
            win_length=None,
            hop_length=512,
            normalized=True
            )
        self.outputs_dir = Path(outputs_dir)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

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
        self.adversarial_waveform, self.animation = self.CW_outter(self.reverse_waveform, self.target)
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
            single channel audio signal sampled at 16kHz

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
        
    def plot_spectrogram(self, waveform, transcription, db_extremes = None):
        """
        Plots a given waveform as an oscillogram and a spectrogram 

        Parameters
        ----------
        waveform : torch.Tensor
            audio signal to be plotted as a spectrogram.
        transcription : string
            transcription of the audio fed into the model.
        db_extremes : list, optional
            hardcoded high and low decibel values used to create a 
            consistent color bar and allow for comparisons.
            The default is None and the values are calculated  
            from the given waveform.

        Returns
        -------
        fig : matplotlib.Pyplot
            the oscillogram and spectrogram of the given waveform.

        """
        spacing = 20
        spectrogram = self.spectrogram_transform(waveform)
        spectrogram_dB = torchaudio.transforms.AmplitudeToDB()(spectrogram)
        if db_extremes:
            db_min = db_extremes[0]
            db_max = db_extremes[1]
        elif db_extremes == None:
            db_min = spectrogram_dB.min()
            db_max = spectrogram_dB.max()
        db_ticks = torch.arange(start = torch.floor(db_min/spacing) * spacing, 
                                end = (torch.ceil(db_max/spacing) + 1) * spacing, 
                                step = spacing)
        duration = waveform.shape[-1]/self.required_sampling_rate
    
        fig = plt.figure(figsize=(15, 8))
    
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(waveform[0])
        ax1.tick_params(bottom = False, labelbottom = False)
    
        # Add dummy axis to top plot to mirror bottom colorbar spacing
        divider_1 = make_axes_locatable(ax1)
        cax1 = divider_1.append_axes("right", size="5%", pad=0.1)
        cax1.axis("off")  # Make it invisible
    
        ax2 = plt.subplot(2, 1, 2)
        im = ax2.imshow(
            spectrogram_dB[0].numpy(),
            aspect="auto",
            origin="lower",
            cmap="magma",
            extent=[
                0, duration,
                0, self.spectrogram_transform.n_fft // 2
            ],
            vmin = db_min,
            vmax = db_max
        )
    
        divider_2 = make_axes_locatable(ax2)
        cax2 = divider_2.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax2, format="%+2.0f dB").set_ticks(db_ticks)
    
        ax2.set_title(f"Spectrogram - Transcription: {transcription}")
        ax2.set_ylabel("Frequency (Hz)")
    
        plt.tight_layout()
    
        return fig
    
    def transcribe(self, waveform):
        """
        Infers upon the Wav2Vec2 model and returns the transcript of a given 
        audio waveform

        Parameters
        ----------
        waveform : torch.Tensor
            audio signal to be input into the Wav2Vec2 model. 

        Returns
        -------
        transcription : string
            transcription of the audio fed into the model.

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
            audio signal to be reversed.

        Returns
        -------
        reverse : torch.Tensor
            reverse of the audio signal given as an input.

        """
        reverse = torch.flip(waveform, dims = [1])
    
        return reverse
    
    def select_target(self):
        """
        Randomly selects a quote from inputs/targets.txt to serve as the 
        target transcription for the upcomming Carlini-Wagner attack

        Returns
        -------
        target : string
            randomly selected quote.

        """
        with open('inputs/targets.txt', mode='r', encoding='utf-8') as file:
            targets = file.read()
            
        targets = targets.split('\n')
        
        random_number_generator = np.random.default_rng()
        target = random_number_generator.choice(targets, size=1)[0].upper()
            
        return target
    
    def CW_outter(self, waveform, target):
        """
        Iteratively adjusts the constant which balances maximizing the
        confidence in the target transrcription while minimizing distortion

        Parameters
        ----------
        waveform : torch.Tensor
            audio signal to be manipulated
        target : string
            transcription the attack aims to produce
        
        Returns
        -------
        best_adverserial_audio : torch.Tensor
            least distorted adverserial audio signal that aims to
            produce the desired transcription
        animation : pathlib.Path
            DESCRIPTION
            
        """
        
        yet_to_converge = True
        c = self.initial_constant
        lowerbound_c = 0
        upperbound_c = 1e9
        best_adverserial_audio = waveform
        best_attack_history = []
        best_transcription_history = []
        for step in range(self.binary_search_steps):
            adverserial_audio, converged, attack_history, transcription_history = self.CW_inner(waveform, target, c)
            if (converged == True):
                yet_to_converge = False
                upperbound_c = c
                c = (lowerbound_c + upperbound_c)/2
                best_adverserial_audio = adverserial_audio
                best_attack_history = attack_history
                best_transcription_history = transcription_history
            elif (converged == False) and (yet_to_converge == False):
                lowerbound_c = c 
                c = (lowerbound_c + upperbound_c)/2
            elif (converged == False) and (yet_to_converge == True):
                lowerbound_c = c 
                c *= 10
        animation = self.animate_attack(best_attack_history, best_transcription_history)
    
        return best_adverserial_audio, animation
                
    def CW_inner(self, waveform, target, c):
        """
        Iteratively builds an distortion pattern which when added
        to the given waveform produces the given traget 
        transcription when fed to the model for a given constant

        Parameters
        ----------
        waveform : torch.Tensor
            audio signal to be manipulated
        target : string
            transcription the attack aims to produce
        c : float
            constant which balances maximizing the confidence 
            in the target transrcription while minimizing distortion
        
        Returns
        -------
        best_adverserial_audio : torch.Tensor
            least distorted adverserial audio signal that aims to
            produce the desired transcription for a given value 
            of c
        converged : boolean
            flag indicating wether the attack successfully converged
        perturbation_history : torch.Tensor
            a list of the adverserial audio signals generated over 
            the course of the attack
        transcription_history : list
            a list of the errant transcriptions produced over the 
            course of the attack
        
        """   
        # 1. Prepare target tokens
        target_tokens = self.processor(text=target, return_tensors="pt").input_ids[0].to(self.device)
        target_length = torch.tensor([len(target_tokens)], dtype=torch.long).to(self.device)
        transcription_history = [''] * self.max_iterations
        dims = list(waveform.size())
        dims[0] = self.max_iterations
        perturbation_history = torch.zeros(dims)
    
        # 2. Normalize the original audio (matching Wav2Vec2's typical input style)
        # Note: If your pipeline handles normalization inside the forward pass, adjust accordingly
        x = waveform.clone().detach().to(self.device)# Shape: [1, seq_len]
        w = torch.atanh((1 - 1e-6) * (waveform)).to(self.device)
    
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
            transcription_history[iteration] = current_transcription
            best_adverserial_example = torch.tanh(w + w_perturbations).detach().cpu()
            perturbation_history[iteration] = best_adverserial_example
    
            progress_bar.set_postfix(
                c=f"{c}",
                Loss=f"{total_loss.item():.4f}",
                L2=f"{loss_distortion.item():.4f}"
            )
    
            # Dynamic scaling (if target is reached, focus more on minimizing distortion)
            if current_transcription == target:
                converged = True
                transcription_history = transcription_history[:iteration + 1]
                perturbation_history = perturbation_history[:iteration + 1]
                print("--> Success! Target achieved. Refining imperceptibility...")
                break
    
        return best_adverserial_example, converged, perturbation_history, transcription_history
    
    def animate_attack(self, attack_history, transcription_history):
        """
        Animates the attack by plotting the osicllograms and spectrograms of
        all of the waveforms along with their respective transcriptions that 
        were generated over the course of the attack

        Parameters
        ----------
        perturbation_history : torch.tensor
            a list of the adverserial audio signals generated over 
            the course of the attack
        transcription_history : list
            a list of the errant transcriptions produced over the 
            course of the attack

        Returns
        -------
        filepath : pathlib.Path
            the path to the resultant gif.

        """
        frames = []
    
        # Find the highest and lowest decibel leves by iterating through all of
        # the waveforms produced over the course of the attack 
        db_extremes = [torch.inf, -torch.inf]
        for attack in attack_history:
            adverserial_waveform = attack.unsqueeze(dim=0)
            spectrogram = self.spectrogram_transform(adverserial_waveform)
            spectrogram_dB = torchaudio.transforms.AmplitudeToDB()(spectrogram)
            local_min = spectrogram_dB.min()
            local_max = spectrogram_dB.max()
            if local_min < db_extremes[0]:
                db_extremes[0] = local_min
            if local_max > db_extremes[1]:
                db_extremes[1] = local_max
        
        plt.ioff()
        
        # Plot each spectrogram and collate each into an animation
        for index in trange(len(transcription_history), desc = 'Visualizing Attack'):
            transcription = transcription_history[index]
            adverserial_waveform = attack_history[index].unsqueeze(dim=0)
            
            spectrogram_figure = self.plot_spectrogram(adverserial_waveform, transcription, db_extremes)
            buf = io.BytesIO()
            spectrogram_figure.savefig(buf, format='jpg', bbox_inches='tight')
            plt.close(spectrogram_figure)
            buf.seek(0)
            with Image.open(buf) as spectrogram:
              frames.append(spectrogram.copy())
            buf.close()
    
        filename = f"{transcription_history[-1]} Carlini-Wagner Spectrogram.gif"
        filepath = self.outputs_dir / filename
        frames[0].save(filepath, save_all = True, 
                       append_images = frames[1:], duration=100, loop=0)
        plt.ion()
          
        return filepath
    
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