# Classification of Emotions in Audio files through CNNs:


 
 **CNNs have proved to be quite efficient and excellent in Computer Vision and Image tasks,  along with transfer learning.**  
 **My main goal is to apply those pattern recognition properties of pretrained Image models, two detect emotions in 2D representation of Audio files, through *(Mel spectrogram, Chromagram , short-time Fourier transform).***
 
***
 ![audio_classifier_img](https://i.ibb.co/GJjcnZP/1-7-Yb-BTqw-F2d-MAu-Qw-Or-D-h-XQ.png)
 

## My Notebooks

***
| Task | Colab
|---|---|
| **`Data Processing : Running yourself`** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1iGp9QXtKr4-WaS6wpXdkAh4LHfurXeto?usp=sharing)|
| **`Model Training : Running yourself`** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KbDsGVhMHo_u7gj33uj9TxIGftQeSCPM?usp=sharing)|
| **`Model Inference (Prediction) : Running yourself`** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1eohYG8gAzRi_6cvMHJ9B9xZ1xyMBcD_P?usp=sharing)|



***
 
***
 
 * **I used 5 pretrained image models, built an ensemble of the 5 models to achieve *third place in this competition, with a final score of 61.0835***
 *  **You can see my solution writeup, data preprocessing techniques,  and model training** [**here**](https://colab.research.google.com/drive/1iGp9QXtKr4-WaS6wpXdkAh4LHfurXeto?usp=sharing)
 
 
### Features to be built for Data preprocessing:
 
 * Here the Idea is to build a 3 channel Image representation of Audio files, where each channel represents a different  form of 2D representation of audio file.

***
 ![img](https://miro.medium.com/max/782/1*rFhL3CYygk0gGlHOmlL_Jg.png)
 
***
 
 *  These three 2D representations will be:
	 *  Mel spectrogram
	 *  Chromagram
	 *  Short-time Fourier transform
* Librosa has a great library for these feature processing. [Librosa lib](https://librosa.org/doc/latest/tutorial.html)

![specs_img](https://i.ibb.co/MM7PGpn/audio-specs.png)

### Miscellaneous features : (Good to have)
* Since I will be using transfer learning, It will much better if we can use pre-trained model that was trained  on audio related tasks, *e.g. We can train our model firstly on our audio dataset in a semi-supervised way* then we can train on our downstream task as a classifier.

### Constraints: 
 * Since Sound is inherently serial, Means sound do not exist as static objects which can be observed in parallel like images, they arrive as sequences of air pressure and meaning about these pressures must be established over time. This sometimes can be tricky to catch and recognize through 2D representations.
 * Although we are converting audio files to 2D image, We can't use same data augmentations techniques. 
 
 * Sounds behave differently with  image style augmentation: e.g. 
	 * Moving a sound event horizontally offsets its position in time, Moving the frequencies of a male voice upwards could change its meaning from man to child.
	 
* [Reference](https://towardsdatascience.com/whats-wrong-with-spectrograms-and-cnns-for-audio-processing-311377d7ccd) 

### Issues with dataset:
* Audio files provided varied in length, from 0.5s to 22s. So, I decided in the competition to use max duration of 5 secs of audio files. 
* Since a big audio file can contain more than one emotion, what we can do is instead of single label classification we use multilabel classification.

***

# [Hackerearth Predict the Emotion Challenge from audio files](https://www.hackerearth.com/challenges/competitive/ia-for-ai/)

![Problem Statement](https://i.ibb.co/Yf1948g/hackimg.png)



## My Bronze Medal 3rd Place Solution Write Up:

* **Leaderboard Rank** : https://www.hackerearth.com/challenges/competitive/ia-for-ai/leaderboard/

![imgage_leaderboard](https://i.ibb.co/wwWnzvM/hackerearth-emotio-detection-leaderboard.png)

* Firstly, I was occupied in my exams didn't have time. I accidently opened hackerearth found the comeptition been going on for about a month.

* I just joined the competition 3 days before the completion.

* So, Being time constrained, I don't time to experiment a lot. First I thought of using torchaudio models. But ran a first draft through it.

* Then I thought, though torchaudio model could give better given time. To do a fast experiment I have to use CNNs, with spectograms.

* I did audio exploration files conatined both mp3 wav. Since I didn't have attened AMA in beginnig didn't find proper meta regarding audio files. I assumed they were sample at traditional 44100 Hz.

* I used a 5 fold Stratified KFold.

* After exploring the audio lengths. They varied from 0.5s to > 22.0s. I took a distibution of lengths of audio files, took a max lenths of 5 secs for audion spectrograms of size (128,455). If files were less than 5.0s, They were extrpolated of to 5 sec using zeros.

* Converted the train and Test files into MelSpecgrams using librosa audio library. Saved on the drive.

* Choose fastai instead of traditional core pytorch. And changed fastai's codes were needed. 

* Experimented with 6 models architectures. 
    * resnest50
    * efficientnet b0-b4.

* There individual best scores varied from 57 to 59
* I tried making ensemble using different weights to 5 models, they didn't gave better results.
* So for final submission and best score of **61.0835**. I simply took a mode of my best submissions of above 6 models.

