"""Build the ten-page research report from the actual experiment outputs."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import src
import json
import numpy as np
import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.utils import ImageReader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results/study'
OUT=ROOT/'output/pdf';OUT.mkdir(parents=True,exist_ok=True)
F=R/'figures'
S=json.loads((R/'summary.json').read_text())
D=pd.read_csv(R/'metrics.csv')
N=D[D.condition=='natural']
C=pd.read_csv(R/'cases.csv')
B=pd.read_csv(R/'museval.csv') if (R/'museval.csv').exists() else None
W=json.loads((R/'worst_learned_case.json').read_text()) if (R/'worst_learned_case.json').exists() else {}
CX=json.loads((R/'context_summary.json').read_text())
CS={s['method']:s for s in CX['summary']}
sensitivity=pd.read_csv(R/'stft_sensitivity.csv')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyProject',fontName='Helvetica',fontSize=10.2,leading=14.3,spaceAfter=8,textColor=colors.HexColor('#22313f')))
styles.add(ParagraphStyle(name='SmallProject',fontName='Helvetica',fontSize=8.5,leading=11.5,spaceAfter=6,textColor=colors.HexColor('#314252')))
styles.add(ParagraphStyle(name='CaptionProject',fontName='Helvetica',fontSize=8.5,leading=11.5,spaceAfter=10,textColor=colors.HexColor('#415566')))
styles.add(ParagraphStyle(name='TitleProject',fontName='Helvetica-Bold',fontSize=22,leading=27,spaceAfter=15,textColor=colors.HexColor('#153b55')))
styles['Heading1'].fontName='Helvetica-Bold';styles['Heading1'].fontSize=16;styles['Heading1'].leading=20
styles['Heading1'].spaceAfter=13;styles['Heading1'].textColor=colors.HexColor('#153b55')
styles['Heading2'].fontSize=11.4;styles['Heading2'].leading=15;styles['Heading2'].spaceBefore=7;styles['Heading2'].spaceAfter=6
flow=[];source=[]
def p(t,small=False):
    flow.append(Paragraph(t,styles['SmallProject' if small else 'BodyProject']));source.append(t)
def h(t):flow.append(Paragraph(t,styles['Heading1']));source.append('\n# '+t)
def sub(t):flow.append(Paragraph(t,styles['Heading2']));source.append('\n## '+t)
def page():flow.append(PageBreak());source.append('\n---\n')
def fig(name,caption,width=490,maxheight=260):
    path=F/name
    w,hh=ImageReader(str(path)).getSize();ratio=min(width/w,maxheight/hh)
    flow.append(Image(str(path),width=w*ratio,height=hh*ratio))
    flow.append(Paragraph(caption,styles['CaptionProject']));source.append(f'![{caption}](../../results/study/figures/{name})')
def eq(tex,label):
    path=OUT/f'eq_{label}.png'
    f=plt.figure(figsize=(7,.46));f.text(.5,.5,'$'+tex+'$',ha='center',va='center',fontsize=12)
    f.savefig(path,dpi=210,bbox_inches='tight',pad_inches=.04,transparent=False);plt.close(f)
    w,hh=ImageReader(str(path)).getSize();ratio=min(475/w,37/hh)
    flow.append(Image(str(path),width=w*ratio,height=hh*ratio));flow.append(Spacer(1,7))
    source.append('$$'+tex+'$$')
def table(rows,widths):
    cells=[[Paragraph(str(v),styles['SmallProject']) for v in row] for row in rows]
    t=Table(cells,colWidths=widths,hAlign='LEFT',repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e9f1f6')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),3),('LINEBELOW',(0,0),(-1,0),.8,colors.HexColor('#71899b')),('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#bccad4'))]))
    flow.append(t);flow.append(Spacer(1,9))
    source.append('\n'.join(' | '.join(map(str,r)) for r in rows))
labels={'mixture':'Mixture','irm':'Power IRM','magnitude_phase':'True magnitude + mixture phase','clipped_psm':'Clipped PSM','nonnegative_psm':'Nonnegative PSM','openunmix':'UMXHQ','htdemucs':'HTDemucs'}
means={row['method']:row for row in S['natural_table']}
def val(m,key='mean_rec_sdr_db'):return means[m][key]

flow.append(Paragraph('Acoustic Limitations of<br/>Vocal Source Separation',styles['TitleProject']))
p('<b>Time-frequency overlap, mixture phase and pretrained reference systems</b>')
p('Jiajun Liu | SID 550378323<br/>ELEC5305 Acoustics, Speech and Signal Processing<br/>8 November 2026',small=True)
sub('Abstract')
p(f'This study asks how overlap and mixture-phase mismatch relate to vocal-separation error. Four oracle reconstructions and two pretrained systems are evaluated on {S["n_excerpts"]} six-second excerpts from {S["n_tracks"]} MUSDB18-HQ test tracks. Natural balances and controlled vocal-to-accompaniment ratios yield {S["n_cases"]} paired inputs. A clipped phase-sensitive mask improves natural-mix reconstruction SDR over a power ideal ratio mask (IRM) by {S["clipped_psm_gain_mean_db"]:.2f} dB on average. An exact coefficient-error identity verifies the phase diagnostic, while the difference after inverse STFT shows why it is not a waveform bound. UMXHQ and HTDemucs provide practical references. The experiment supports a conditional account of separation difficulty, rather than a universal ranking or a subjective quality claim.')
h('1. Introduction and research question')
p('Vocal extraction supports karaoke, remixing and analysis, but musical sources can share harmonics, transients and stereo positions. A stronger model can exploit learned regularities; it cannot make overlap and destructive interference irrelevant. Comparing a hand-designed mask with a neural network alone obscures this issue, because many neural separators also estimate spectral representations.')
p('<b>Research question:</b> How do time-frequency overlap and mixture-phase mismatch constrain oracle magnitude-mask reconstruction, and do two pretrained vocal separators exhibit similar difficulty trends under controlled accompaniment gain changes?')
p('The project reproduces the original power-IRM and true-magnitude baselines, adds two established phase-sensitive oracle variants, and compares them on matched inputs. The contribution is the controlled musical analysis and verified implementation, not the invention of phase-sensitive masking or a newly trained network. The scope is deliberately limited to short excerpts and a small test subset.')
fig('pipeline.png','Figure 1. Study sequence. Ground-truth stems are withheld from the learned systems.',maxheight=110)
sub('Contents')
toc_rows = [
    ['1 Introduction and question - p. 1', '6 Natural-mixture results - p. 6'],
    ['2 Literature review - p. 2', '7 Controlled balance and phase - p. 7'],
    ['3 Data and paired conditions - p. 3', '8 Failure, context and sensitivity - p. 8'],
    ['4 Oracle constraints and modification - p. 4', '9 Discussion and limitations - p. 9'],
    ['5 Evaluation and implementation - p. 5', '10 Conclusion and references - p. 10'],
]
toc = Table([[Paragraph(t,styles['SmallProject']) for t in row] for row in toc_rows], colWidths=[245,245], hAlign='LEFT')
toc.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),0),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),0),('BOTTOMPADDING',(0,0),(-1,-1),0)]))
flow.append(toc)
source.append('\n'.join(' | '.join(row) for row in toc_rows))


page();h('2. Literature review and rationale')
sub('2.1 Magnitude masks and phase-sensitive objectives')
p('Wang and Chen [1] organize supervised separation around targets such as binary masks, ratio masks, spectral magnitudes and phase-sensitive masks. These targets impose different constraints; the term IRM also has multiple exponent conventions. This project fixes the power ratio with exponent one. Its appeal is a smooth, bounded attenuation mask, but its source-power denominator omits the instantaneous cross term in mixture power. Therefore an oracle power mask need not minimize the error of a particular mixture realization.')
p('Erdogan et al. [2] show that using mixture phase changes the amplitude that minimizes complex-spectrum error. Matching the true target magnitude is consequently not always optimal. Their work motivates projecting the target coefficient onto the mixture direction. Here the projection is implemented directly from known music stems, first clipped to [0,1], then allowed nonnegative amplification. This reproduces the masking idea, not their trained speech recognizer or recurrent-network experiments. The comparison tests which restriction matters in music.')
sub('2.2 Learned spectral and hybrid reference systems')
p('Open-Unmix [3] provides an accessible, reproducible separation reference. Its bidirectional recurrent magnitude estimator and multichannel Wiener reconstruction [4] supply a useful learned spectral contrast. The present UMXHQ configuration predicts all four stems and uses one Wiener iteration. Calling it a fixed mixture-phase mask would be inaccurate: its multichannel reconstruction is more general than the scalar oracle masks used here. Short excerpts also remove context available in full-track inference.')
p('HTDemucs [5] exchanges information between waveform and spectral branches through attention. It is selected to test a different representation and context mechanism. The paper distinguishes architecture, additional training data and fine-tuning effects; these factors prevent attributing every gain to phase modeling. In particular, this is not a matched-training-data architecture comparison. This study uses the official htdemucs checkpoint [6], trained on MUSDB plus 800 extra songs, without fine-tuning or random shift averaging. It is a reference configuration, not a reproduction of the paper\'s strongest published result.')
p('Later complex-spectral systems such as BS-RoFormer [7] further undermine a simple "spectral versus deep learning" distinction. Its band-wise attention and complex representation suggest a useful future reference, but it is not evaluated here. Neither selected model is presented as the current state of the art.')
sub('2.3 Measurement and the local research gap')
p('Le Roux et al. [8] explain why SDR definitions and scale conventions can produce misleading comparisons. This motivates reporting reconstruction SDR, explicitly defined stereo SI-SDR and separate BSS Eval v4 scores. The local question is not whether a published method is globally superior, but whether measurable acoustic variables help explain errors on paired examples. The experiment joins interpretable oracle constraints with fixed pretrained systems; it does not claim that this question has never been studied.')
table([['Approach','Useful property','Key limitation'],['Power IRM / phase oracles','Known-reference diagnosis','Unavailable without stems'],['UMXHQ','Reproducible spectral reference','Learned priors; context dependent'],['HTDemucs','Hybrid waveform/spectral processing','Different architecture and training recipe']],[135,170,185])

page();h('3. Methods: data and paired conditions')
sub('3.1 Deterministic test subset')
p('MUSDB18-HQ supplies 44.1 kHz stereo mixtures and vocal, drum, bass and other stems [9]. The lexicographically first five official test tracks are selected before inspecting separation scores. Two fixed excerpts, starting at 30 and 60 seconds and lasting six seconds each, give ten candidate excerpts. A vocal RMS threshold of -50 dBFS rejects effectively silent targets; exclusions and their reasons are saved in the run manifest. This is a convenience subset, not a representative random sample of music.')
table([['Official test track','Excerpt intervals (s)']]+[[t,'30-36; 60-66'] for t in json.loads((R/'run_manifest.json').read_text())['tracks']],[345,145])
p('The 60-second Angels In Amplifiers excerpt is excluded by the vocal-activity rule, leaving nine active excerpts. Official-archive WAVs are verified by uncompressed length and CRC32. No training track is used in the reported evaluation; the subset manifest records provenance.')
sub('3.2 Exact mixtures and gain intervention')
p('For each stereo excerpt, v denotes vocals and a is drums + bass + other. Experimental inputs are reconstructed as x = v + ga for every system. This avoids treating independently quantized mixture files as exactly additive. The original file-versus-stem-sum relative discrepancy is measured separately; its maximum is '+f'{S["max_original_mixture_relative_error"]:.2e}'+'. Scores use the unscaled true vocal reference.')
eq(r'\mathrm{VAR}=10\log_{10}\frac{\sum_{n,c}v_c[n]^2}{\sum_{n,c}a_c[n]^2},\qquad g=10^{(\mathrm{VAR}_{native}-\mathrm{VAR}_{target})/20}','var')
p(f'Each base excerpt is used at its native balance and at -10, 0 and +10 dB target VAR. Only the accompaniment amplitude changes. The resulting {S["n_cases"]} inputs are paired across all seven methods. Repeated mixtures of one excerpt are not treated as independent songs. VAR, overlap and phase mismatch change together; gain is the controlled intervention, whereas their individual causal effects are not separately identified.')
sub('3.3 Shared analysis transform')
p('The analysis uses a periodic Hann window, a 2048-point FFT and a 512-sample hop, with centred constant padding. Stereo channels are transformed separately and inverted to the exact original length. Natural excerpts are additionally processed at 1024/256 and 4096/1024 FFT/hop settings to test representation sensitivity. Learned models retain their author-defined native transforms; changing them would invalidate the pretrained weights.')

page();h('4. Methods: oracle constraints and the modification')
sub('4.1 Reproduced baselines')
eq(r'X=V+A,\qquad M_{IRM}=\frac{|V|^2}{|V|^2+|A|^2},\qquad \widehat V_{IRM}=M_{IRM}X','irm')
p('All coefficients are per channel, frequency and frame. Bins with zero total source power receive mask zero. The original power-IRM implementation uses true source powers, so it measures a representation-and-mask reference rather than blind separation performance. The input mixture itself is included as a no-separation control.')
eq(r'\widehat V_{mag}=|V|e^{j\angle X},\quad D_{\phi}=\frac{\sum |V|^2[1-\cos(\angle V-\angle X)]}{\sum |V|^2}','phase')
p('The second baseline keeps the exact target magnitude and replaces target phase with mixture phase. It isolates the coefficient error of that substitution, but is not the best possible waveform reconstruction achievable by every magnitude-based model.')
sub('4.2 Small modification: phase-sensitive projection')
eq(r'm^*=\frac{\operatorname{Re}(V X^*)}{|X|^2}=\frac{|V|}{|X|}\cos(\angle V-\angle X)','psm')
p('Minimizing |mX - V| squared over real m gives this projection [2]. Clipping m* to [0,1] gives the optimum within the ordinary attenuation-mask interval. A second version clips only below zero, allowing amplification. Zero-mixture bins receive mask zero. In every case the estimate is mX; no true phase is inserted during synthesis.')
p('This modification uses the same signals and STFT as the reproduced IRM. Its purpose is to separate errors due to source-power masking from errors imposed by the nonnegative mixture-direction constraint. It is an oracle extension of the experiment. It cannot be applied to an unknown song without estimating the missing target information.')
sub('4.3 Why magnitude accuracy is insufficient')
eq(r'\frac{\sum|\widehat V_{mag}-V|^2}{\sum|V|^2}=2D_{\phi}','identity')
p('This coefficient identity follows by expanding the squared distance between two equal-length complex vectors. It is checked numerically. A phase-sensitive amplitude can shorten a badly oriented coefficient and reduce its error, even though its magnitude becomes less accurate. When cancellation makes |X| smaller than |V|, a mask capped at one cannot restore the lost magnitude. When the target points against X, a nonnegative mask cannot reverse direction.')
p('These statements concern coefficient geometry. Overlapping STFT coefficients are redundant and modified spectra need not be consistent with a waveform. Inverse STFT projects them back to an audio signal, so coefficient-wise optimality is not a universal guarantee of highest waveform SDR. The paired measured results are needed to establish the actual waveform effect.')

page();h('5. Methods: difficulty, evaluation and implementation')
sub('5.1 Acoustic difficulty and waveform metrics')
eq(r'O=\frac{\sum\min(|V|^2,|A|^2)}{\sum|V|^2},\qquad 0\leq O\leq1','overlap')
p('O measures accompaniment energy occupying vocal-energy bins. It depends on relative levels as well as spectral occupancy; it is not a level-invariant overlap index. D_phi ranges from zero to two and weights phase mismatch by vocal energy. Both descriptors use all channels and bins of the same analysis STFT. Natural-mix scatterplots and descriptive Spearman correlations complement the controlled gain sweep.')
eq(r'\mathrm{SDR}_{rec}=10\log_{10}\frac{\|v\|^2}{\|v-\widehat v\|^2}','sdr')
p('Reconstruction SDR fits no gain, delay or distortion filter. It penalizes vocal attenuation and interference together. SI-SDR first demeans each channel, concatenates the two channels, and fits one shared scalar projection onto the reference. The same operation is used for every method. This is an explicitly stated stereo extension of SI-SDR [8]; it is not the average of two independently rescaled mono scores.')
p('BSS Eval v4 [10] separately evaluates two source images: vocals and accompaniment, with estimated accompaniment defined as mixture minus estimated vocals. It uses one-second nonoverlapping score windows, 512-tap distortion filters fixed across each excerpt, no permutation search and the image formulation. The mixture control is omitted here because its exactly zero accompaniment estimate is rejected by museval; no artificial noise is added. The median valid vocal SDR across windows is retained. These short-excerpt settings differ from whole-track benchmark aggregation, so published benchmark numbers are not directly comparable.')
sub('5.2 Fair inference and aggregation')
p('UMXHQ uses four target networks, its native 4096-point STFT / 1024 hop and one multichannel Wiener iteration. HTDemucs uses the htdemucs checkpoint with the official shared mean/standard-deviation preprocessing, split inference and 25% overlap. Random time shifts are disabled. Both run on CPU with four Torch threads. A targeted iteration gives both models 18-second inputs with six seconds of context on each side, scores only the original centre six seconds, and repeats this for all nine natural excerpts. No model is retrained or fine-tuned.')
p('Scores are computed before playback gain. Natural-mix scores are averaged over the retained excerpts within each track, then equally across tracks. Each track retains at most two excerpts. Track dots expose variability; controlled curves use the same complete paired set at every VAR. Correlations are descriptive and have no independent-sample p-values. Model timing excludes initial download and loading, and is not a general hardware benchmark.')
sub('5.3 Reproducibility and checks')
p('The pipeline reads only requested WAV frames, checks sample rate, length and channel alignment, and rejects invalid intervals. Checkpoints bind each model output to a code signature and input hash. CSVs retain per-case scores, acoustic variables, gains and runtime; manifests retain package versions and settings. Thirteen automated tests cover reconstruction, phase identities, cancellation, zero bins, scale conventions and stem alignment. The maximum measured STFT relative reconstruction error is '+f'{S["max_stft_relative_error"]:.2e}'+'. WAV comparisons share a single gain per excerpt and use floating-point storage.')

page();h('6. Results: natural musical mixtures')
fig('natural_scores.png','Figure 2. Natural balances. Bars equally weight tracks after averaging the retained excerpts; dots show track means. Oracle and learned methods have different information access.',maxheight=230)
rows=[['Method','Rec. SDR (dB)','SI-SDR (dB)','Track range (rec.)']]
for m in labels:
    if m not in means:continue
    z=means[m];rows.append([labels[m],f'{z["mean_rec_sdr_db"]:.2f}',f'{z["mean_si_sdr_db"]:.2f}',f'{z["min_track_rec_sdr_db"]:.2f} to {z["max_track_rec_sdr_db"]:.2f}'])
table(rows,[215,85,85,105])
p('Table 1. Means over tracks at the native mix balance; ranges are minimum and maximum track means, not confidence intervals.',small=True)
p(f'The reproduced power IRM reaches {val("irm"):.2f} dB mean reconstruction SDR. Clipped PSM reaches {val("clipped_psm"):.2f} dB, a paired improvement of {S["clipped_psm_gain_mean_db"]:.2f} dB. Across the retained excerpts the gain ranges from {S["clipped_psm_gain_min_db"]:.2f} to {S["clipped_psm_gain_max_db"]:.2f} dB. Allowing nonnegative amplification gives {val("nonnegative_psm"):.2f} dB. These values support a measurable cost of the original mask constraints on this subset.')
p(f'The true-magnitude/mixture-phase reconstruction obtains {val("magnitude_phase"):.2f} dB. Its ordering relative to the projection masks is consistent with the fact that exact magnitude is not the coefficient-error optimum when phase is wrong. UMXHQ and HTDemucs obtain {val("openunmix"):.2f} and {val("htdemucs"):.2f} dB respectively. Their gap includes architecture, priors and reconstruction differences; the data do not isolate any one cause.')
if B is not None:
    b=B.groupby(['track','method']).bss_eval_v4_vocal_sdr_db.mean().groupby('method').mean()
    p('The separate median-window BSS Eval v4 cross-check gives '+', '.join(f'{labels[k]} {b[k]:.2f} dB' for k in ['irm','clipped_psm','openunmix','htdemucs'] if k in b)+'. Differences from Table 1 reflect window aggregation and metric definitions, rather than another separation run.',small=True)

page();h('7. Results: controlled balance and phase error')
fig('var_sweep.png','Figure 3. Same excerpts remixed at three target VARs. Points are paired averages; curves do not identify independent effects of overlap and phase.',maxheight=220)
controlled=D[D.condition!='natural'].copy();controlled['target_var']=controlled.var_db.round()
ct=controlled.groupby(['target_var','method']).reconstruction_sdr_db.mean().unstack()
p('The controlled inputs preserve musical content and alignment while changing accompaniment strength. The no-separation control equals target VAR by construction, providing a direct check of the gain implementation. Actual measured means at -10, 0 and +10 dB are '+', '.join(f'{ct.loc[z,"mixture"]:.2f}' for z in [-10.,0.,10.])+' dB.')
p('For IRM the corresponding reconstruction scores are '+', '.join(f'{ct.loc[z,"irm"]:.2f}' for z in [-10.,0.,10.])+' dB; for clipped PSM they are '+', '.join(f'{ct.loc[z,"clipped_psm"]:.2f}' for z in [-10.,0.,10.])+' dB. The paired design establishes sensitivity to accompaniment gain. It cannot tell whether overlap or phase mismatch alone caused a model\'s degradation, since both change when the same accompaniment is scaled.')
fig('difficulty.png','Figure 4. Natural-excerpt relationships. Each point is an excerpt, with repeated excerpts within tracks. Correlations are descriptive; acoustic variables are coupled.',maxheight=190)
p('The scatterplots retain the natural content variability rather than combining all remixes into an apparently large independent sample. The maximum absolute residual of the coefficient identity is '+f'{S["max_phase_identity_error"]:.2e}'+'. This verifies the calculation of D_phi and the true-magnitude oracle. The identity is partly definitional, so its verification is a correctness check rather than new empirical evidence of a causal relation. Waveform results carry the substantive test.')

page();h('8. Results: failure inspection and sensitivity')
fig('failure_spectrograms.png','Figure 5. The natural excerpt with the lowest learned-system SI-SDR. All panels share one magnitude reference and colour scale. The selection rule is fixed and stated in failure_case.txt.',maxheight=385)
p(f'A common scale avoids hiding attenuation by per-panel normalization. In {W.get("track","the selected excerpt")} at {W.get("start_sec",0):g} s, {labels.get(W.get("method"),"the learned system")} gives {W.get("reconstruction_sdr_db",0):.2f} dB reconstruction SDR but {W.get("si_sdr_db",0):.2f} dB SI-SDR. Its demeaned output/reference RMS ratio is {W.get("demeaned_rms_ratio",0):.4f}. The very weak, poorly aligned estimate approaches the zero-output reconstruction error, explaining why a near-zero reconstruction SDR can coexist with very low SI-SDR. This is an observed failure case, not a listening rating or a model-wide characterization.')
p(f'The targeted context iteration raises this failed-case HTDemucs score from {CX["failed_case"]["short_reconstruction_sdr_db"]:.2f} to {CX["failed_case"]["reconstruction_sdr_db"]:.2f} dB. Across all nine natural excerpts, equal-track mean gains are {CS["openunmix"]["track_mean_rec_gain_db"]:.2f} dB for UMXHQ and {CS["htdemucs"]["track_mean_rec_gain_db"]:.2f} dB for HTDemucs. Not every excerpt improves. Added context and Demucs split-window alignment change together, so this is an inference-configuration improvement, not proof of a single cause.')
ss=sensitivity.groupby(['n_fft','method']).reconstruction_sdr_db.mean().unstack()
p('The STFT sensitivity experiment gives IRM means of '+', '.join(f'{ss.loc[n,"irm"]:.2f}' for n in [1024,2048,4096])+' dB for 1024, 2048 and 4096 points respectively. The complete plot is supplied with the results. Frequency resolution, transient smearing and overlap change together. The sweep tests robustness of the interpretation; the best setting on these excerpts is not claimed to be generally optimal.')
p('Reference, mixture, oracle and learned WAVs are available for the first excerpt of every track with one shared playback gain. The video includes fixed excerpts from the first selected track. These sound examples support informal inspection and demonstration, not population-level preference claims.',small=True)

page();h('9. Discussion: what the experiments establish')
sub('9.1 Interpreting the modification')
p('The phase-sensitive comparison makes the failure mechanism more specific than "the sources overlap." Power IRM uses source energies but ignores instantaneous relative phase. The constrained projection uses the direction of the actual mixture coefficient. Its measured gain shows that part of the IRM error can be reduced without directly replacing mixture phase. This is consistent with the amplitude-phase interaction in [2], while extending the local demonstration to stereo musical stems. It is not evidence for an algorithmically novel separator.')
p('The nonnegative unbounded projection isolates a second restriction: attenuation-only masks cannot increase a cancelled mixture coefficient. Its additional gain points to the cost of the upper mask cap. A remaining error still exists because a nonnegative multiplier cannot rotate a coefficient to an arbitrary target phase. Thus a larger magnitude error can coexist with a smaller complex or waveform error. This is why "true magnitude plus mixture phase" must not be called an absolute oracle ceiling for all learned systems.')
sub('9.2 Relating oracle and learned behavior')
p('A learned separator infers target structure from data rather than observing true stems. Overlap and phase descriptors therefore cannot capture every source of error: timbre, arrangement, stereo image, training distribution and excerpt boundaries matter. UMXHQ adds multichannel Wiener reconstruction, and HTDemucs combines temporal and spectral pathways. Neither fits the same scalar-mask constraint as the baseline. A learned model exceeding IRM on an example would not contradict the oracle calculations; falling below IRM would not identify phase as the sole bottleneck.')
p('The controlled gain sweep answers a narrower causal question: how outputs change when accompaniment strength changes for fixed content. Correlated increases in O and D_phi describe this intervention. Natural-mix correlations additionally include content changes and cannot establish independent physical causes. The phase identity shares terms with its diagnostic by construction; reporting it as a predictive discovery would overstate the experiment.')
sub('9.3 Limitations, criticism and next experiments')
p(f'Five alphabetically selected tracks and two fixed candidate six-second intervals provide a reproducible case study, not a random or comprehensive benchmark. The repeated remixes represent only {S["n_excerpts"]} retained excerpts from five tracks. There are no justified population confidence intervals, model-significance claims or genre-general conclusions. A stronger replication would predefine a larger stratified test sample and use track-level uncertainty estimates. Silent-vocal segments would require separate false-positive measures rather than an undefined target SDR.')
p('Short inference windows remove full-song context, especially relevant to the bidirectional UMX network. The measured context iteration supports testing inference boundaries before blaming all failures on overlap or phase. Its heterogeneous effects argue against a universal context gain. A full-track control and fixed split alignment would better separate these factors. Phase statistics remain representation dependent, as the STFT sweep illustrates.')
p('The default HTDemucs checkpoint used 800 additional training songs, while UMXHQ did not use that same extra corpus. Architecture-only superiority cannot be inferred from this comparison. Signal-distance scores cannot establish intelligibility, absence of musical noise or listener preference. A blinded, level-matched listening comparison would be needed. Further work should also separate level from spectral overlap, compare complex-mask references, and evaluate a contemporary band-split model under explicitly matched training-data and metric conditions. Starting again, the most valuable early step would be a verified one-track pipeline and a fixed manifest, before expanding the model list.')

page();h('10. Conclusion')
p(f'The project produced a reproducible vocal-separation investigation with {S["n_cases"]} paired musical inputs, two pretrained references and explicit signal-level evaluation. On native mixtures, clipping the phase-sensitive projection improves on power IRM by {S["clipped_psm_gain_mean_db"]:.2f} dB mean reconstruction SDR. The verified phase identity and the amplification ablation explain why exact target magnitude with mixture phase is neither a guarantee of optimal reconstruction nor a universal learned-model ceiling. Controlled gain changes expose conditional difficulty trends, while the small subset, coupled descriptors and short context limit generalization. A targeted context iteration also improves a short-input failure without retraining. The conclusions remain conditional on these examples, with broader acoustic and perceptual validation still needed.')
h('References')
refs=[
('[1]','Wang D, Chen J. Supervised speech separation based on deep learning: an overview. IEEE/ACM TASLP. 2018;26(10):1702-1726.','https://doi.org/10.1109/TASLP.2018.2842159'),
('[2]','Erdogan H, Hershey JR, Watanabe S, Le Roux J. Phase-sensitive and recognition-boosted speech separation using deep recurrent neural networks. ICASSP. 2015.','https://www.jonathanleroux.org/pdf/Erdogan2015ICASSP04.pdf'),
('[3]','Stoter FR, Uhlich S, Liutkus A, Mitsufuji Y. Open-Unmix: a reference implementation for music source separation. JOSS. 2019;4(41):1667.','https://doi.org/10.21105/joss.01667'),
('[4]','SigSep. Open-Unmix technical documentation and PyTorch implementation, version 1.3.0. Accessed 8 November 2026.','https://github.com/sigsep/open-unmix-pytorch'),
('[5]','Rouard S, Massa F, Defossez A. Hybrid transformers for music source separation. ICASSP. 2023. Preprint arXiv:2211.08553.','https://arxiv.org/abs/2211.08553'),
('[6]','Meta AI. Demucs author implementation, version 4.0.1; htdemucs checkpoint 955717e8-8726e21a. Accessed 8 November 2026.','https://github.com/facebookresearch/demucs'),
('[7]','Lu WT, Wang JC, Kong Q, Hung YN. Music source separation with band-split RoPE transformer. arXiv:2309.02612v2. 2023.','https://arxiv.org/abs/2309.02612'),
('[8]','Le Roux J, Wisdom S, Erdogan H, Hershey JR. SDR: half-baked or well done? ICASSP. 2019:626-630.','https://arxiv.org/abs/1811.02508'),
('[9]','Rafii Z, Liutkus A, Stoter FR, Mimilakis SI, Bittner R. MUSDB18-HQ: an uncompressed version of MUSDB18. Zenodo. 2019.','https://doi.org/10.5281/zenodo.3338373'),
('[10]','SigSep. museval: BSS Eval v4 implementation, version 0.4.1. Accessed 8 November 2026.','https://github.com/sigsep/sigsep-mus-eval')]
for number,text,url in refs:
    p(f'{number} {text} <link href="{url}" color="#245f89">Source</link>.',small=True)
sub('Code, data and assistance')
p('Code and reproduction instructions accompany this report. The existing project repository is <link href="https://github.com/JiajunLiu-June/elec5305-project-530878323" color="#245f89">JiajunLiu-June/elec5305-project-530878323</link>. Original course project work was extended with AI-assisted implementation, analysis and report drafting; this assistance is acknowledged. Model authors retain credit for pretrained systems, and phase-sensitive masking is attributed to prior work. No formal listener ratings or personal listening observations are fabricated. Source audio is obtained separately under MUSDB18-HQ terms, rather than included in the GitHub source archive.',small=True)

def furniture(c,doc):
    c.setStrokeColor(colors.HexColor('#cbd8e1'));c.line(52,805,543,805)
    c.setFont('Helvetica',8);c.setFillColor(colors.HexColor('#5a6e7c'))
    c.drawString(52,815,'ELEC5305 | Vocal source separation')
    c.drawRightString(543,815,'Jiajun Liu | 550378323')
    c.drawString(52,28,'Research report | 8 November 2026')
    c.drawRightString(543,28,str(doc.page))

path=OUT/'ELEC5305_Project_Report_Jiajun_Liu.pdf'
doc=SimpleDocTemplate(str(path),pagesize=(595.28,841.89),rightMargin=52,leftMargin=52,topMargin=52,bottomMargin=47,title='Acoustic Limitations of Vocal Source Separation',author='Jiajun Liu')
doc.build(flow,onFirstPage=furniture,onLaterPages=furniture)
(OUT/'report_source.md').write_text('\n\n'.join(source))
from pypdf import PdfReader
pages=len(PdfReader(path).pages)
print(path, 'pages', pages)
if pages!=10:raise RuntimeError(f'Expected 10 pages, got {pages}; inspect and repair layout')
