# ELEC5305 Project – Music Source Separation

## Project Title

**Music Source Separation: A Comparative Study of Time-Frequency Masking and Deep Learning for Vocal Separation**

## Project Overview

This project investigates music source separation, with a primary focus on separating vocals from instrumental accompaniment in mixed music recordings.

The project will explore and compare traditional signal-processing-based source separation techniques with deep-learning-based approaches. In particular, time-frequency representations such as the Short-Time Fourier Transform (STFT) will be used to analyse mixed audio signals and construct masks for separating different sound sources.

The initial scope of the project is vocal/accompaniment separation. If time permits, the system may be extended to multi-source separation, such as separating vocals, drums, bass, and other instruments.

## Objectives

The main objectives of this project are:

- Analyse music signals in the time-frequency domain.
- Implement a baseline vocal separation method using time-frequency masking.
- Investigate a deep-learning-based source separation approach.
- Compare the performance of different separation methods.
- Evaluate both objective separation quality and perceptual audio quality.
- Demonstrate the system using real music examples.

## Proposed Methodology

The project is expected to involve:

- Audio preprocessing
- Short-Time Fourier Transform (STFT)
- Spectrogram analysis
- Time-frequency masking
- Inverse STFT for audio reconstruction
- Deep-learning-based source separation
- Quantitative and qualitative performance evaluation

The main implementation platform is expected to be **Python**, with appropriate audio and signal-processing libraries.

## Evaluation

The performance of the separation methods will be evaluated using suitable source-separation metrics, such as:

- Signal-to-Distortion Ratio (SDR)
- Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)
- Spectral comparison
- Listening-based qualitative evaluation

The project will compare the baseline method with an improved or deep-learning-based method to investigate the strengths and limitations of different approaches.

## Expected Outcome

The expected outcome is a working music source separation prototype that can take a mixed music recording as input and produce separated vocal and accompaniment audio tracks.

The final project will include:

- Source code
- Example input and separated audio outputs
- Experimental results
- Performance comparison
- Project report
- Video demonstration

## Timeline (Weeks 1–13)

| Weeks | Task |
|---|---|
| 1–2 | Project topic selection and scope definition. Define music source separation as the project area and establish vocal/accompaniment separation as the initial target. |
| 3–5 | Conduct a literature review on music source separation, STFT, time-frequency masking, and deep-learning-based approaches. Select a suitable dataset and evaluation metrics. |
| 6–7 | Implement the baseline system, including audio preprocessing, STFT analysis, time-frequency masking, and inverse STFT reconstruction. |
| 8–9 | Implement or integrate a deep-learning-based source separation approach and perform initial experiments on the selected dataset. |
| 10–11 | Optimise and evaluate the separation methods using metrics such as SDR and SI-SDR, together with spectral and listening-based comparisons. Analyse failure cases and improve the implementation where appropriate. |
| 12 | Conduct final experiments, organise quantitative results, spectrograms, comparison figures, and representative audio examples. |
| 13 | Complete the final report, GitHub documentation, and video demonstration. |

## Repository

This repository contains the work for the **ELEC5305 Speech and Audio Processing Project**.

Additional implementation details, experimental results, and documentation will be added throughout the project.
