## Classification of Emotions in Audio files through CNNs:
 
 **CNNs have proved to be quite efficient and excellent in Computer Vision and Image tasks,  along with transfer learning.**  
 **My main goal is to apply those pattern recognition properties of pretrained Image models, two detect emotions in 2D representation of Audio files, through *(Mel spectrogram, Chromagram , short-time Fourier transform).***
 ![img](https://miro.medium.com/max/782/1*rFhL3CYygk0gGlHOmlL_Jg.png)
 
 * **I used 5 pretrained image models, built an ensemble of the 5 models to achieve *third place in this competition, with a final score of 61.0835***
 *  **You can see my solution writeup, data preprocessing techniques,  and model training** [**here**](https://colab.research.google.com/drive/1iGp9QXtKr4-WaS6wpXdkAh4LHfurXeto?usp=sharing)
 
 
### Features to be built for Data preprocessing:
 
 * Here the Idea is to build a 3 channel Image representation of Audio files, where each channel represents a different  form of 2D representation of audio file.
 *  These three 2D representations will be:
	 *  Mel spectrogram
	 *  Chromagram
	 *  Short-time Fourier transform
* Librosa has a great library for these feature processing. [Librosa lib](https://librosa.org/doc/latest/tutorial.html)

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


<a href="https://imgbb.com/"><img src="https://i.ibb.co/VV3tzpH/1-r-Fh-L3-CYygk0g-Gl-HOml-L-Jg.png" alt="1-r-Fh-L3-CYygk0g-Gl-HOml-L-Jg" border="0"></a>
